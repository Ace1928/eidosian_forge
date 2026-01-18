import copy
from packaging import version as packaging_version  # pylint: disable=g-bad-import-order
import os.path
import re
import sys
from google.protobuf.any_pb2 import Any
from google.protobuf import text_format
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import context
from tensorflow.python.framework import byte_swap_tensor as bst
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import importer
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import versions
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
def import_scoped_meta_graph_with_return_elements(meta_graph_or_file, clear_devices=False, graph=None, import_scope=None, input_map=None, unbound_inputs_col_name='unbound_inputs', restore_collections_predicate=lambda key: True, return_elements=None):
    """Imports graph from `MetaGraphDef` and returns vars and return elements.

  This function takes a `MetaGraphDef` protocol buffer as input. If
  the argument is a file containing a `MetaGraphDef` protocol buffer ,
  it constructs a protocol buffer from the file content. The function
  then adds all the nodes from the `graph_def` field to the
  current graph, recreates the desired collections, and returns a dictionary of
  all the Variables imported into the name scope.

  In combination with `export_scoped_meta_graph()`, this function can be used to

  * Serialize a graph along with other Python objects such as `QueueRunner`,
    `Variable` into a `MetaGraphDef`.

  * Restart training from a saved graph and checkpoints.

  * Run inference from a saved graph and checkpoints.

  Args:
    meta_graph_or_file: `MetaGraphDef` protocol buffer or filename (including
      the path) containing a `MetaGraphDef`.
    clear_devices: Boolean which controls whether to clear device information
      from graph_def. Default false.
    graph: The `Graph` to import into. If `None`, use the default graph.
    import_scope: Optional `string`. Name scope into which to import the
      subgraph. If `None`, the graph is imported to the root name scope.
    input_map: A dictionary mapping input names (as strings) in `graph_def` to
      `Tensor` objects. The values of the named input tensors in the imported
      graph will be re-mapped to the respective `Tensor` values.
    unbound_inputs_col_name: Collection name for looking up unbound inputs.
    restore_collections_predicate: a predicate on collection names. A collection
      named c (i.e whose key is c) will be restored iff
      1) `restore_collections_predicate(c)` is True, and
      2) `c != unbound_inputs_col_name`.
    return_elements:  A list of strings containing operation names in the
      `MetaGraphDef` that will be returned as `Operation` objects; and/or
      tensor names in `MetaGraphDef` that will be returned as `Tensor` objects.

  Returns:
    A tuple of (
      dictionary of all the `Variables` imported into the name scope,
      list of `Operation` or `Tensor` objects from the `return_elements` list).

  Raises:
    ValueError: If the graph_def contains unbound inputs.

  """
    if context.executing_eagerly():
        raise ValueError('Exporting/importing meta graphs is not supported when eager execution is enabled.')
    if isinstance(meta_graph_or_file, meta_graph_pb2.MetaGraphDef):
        meta_graph_def = meta_graph_or_file
    else:
        meta_graph_def = read_meta_graph_file(meta_graph_or_file)
    if unbound_inputs_col_name:
        for key, col_def in meta_graph_def.collection_def.items():
            if key == unbound_inputs_col_name:
                kind = col_def.WhichOneof('kind')
                field = getattr(col_def, kind)
                if field.value and (not input_map or sorted([compat.as_str(v) for v in field.value]) != sorted(input_map)):
                    raise ValueError('Graph contains unbound inputs: %s. Must provide these inputs through input_map.' % ','.join((compat.as_str(v) for v in field.value if not input_map or v not in input_map)))
                break
    graph = graph or ops.get_default_graph()
    with graph.as_default():
        producer_op_list = None
        if meta_graph_def.meta_info_def.HasField('stripped_op_list'):
            producer_op_list = meta_graph_def.meta_info_def.stripped_op_list
        input_graph_def = meta_graph_def.graph_def
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ''
        scope_to_prepend_to_names = graph.unique_name(import_scope or '', mark_as_used=False)
        imported_return_elements = importer.import_graph_def(input_graph_def, name=import_scope or scope_to_prepend_to_names, input_map=input_map, producer_op_list=producer_op_list, return_elements=return_elements)
        tf_version = meta_graph_def.meta_info_def.tensorflow_version
        if not tf_version:
            variables_have_trainable = True
        else:
            variables_have_trainable = packaging_version.parse(tf_version) >= packaging_version.parse('1.9')
        sorted_collections = []
        if ops.GraphKeys.TRAINABLE_VARIABLES in meta_graph_def.collection_def:
            sorted_collections.append((ops.GraphKeys.TRAINABLE_VARIABLES, meta_graph_def.collection_def[ops.GraphKeys.TRAINABLE_VARIABLES]))
        for key, value in sorted(meta_graph_def.collection_def.items()):
            if key != ops.GraphKeys.TRAINABLE_VARIABLES:
                sorted_collections.append((key, value))
        variable_objects = {}
        for key, col_def in sorted_collections:
            if key == unbound_inputs_col_name:
                continue
            if not restore_collections_predicate(key):
                continue
            kind = col_def.WhichOneof('kind')
            if kind is None:
                logging.error('Cannot identify data type for collection %s. Skipping.', key)
                continue
            from_proto = ops.get_from_proto_function(key)
            if key == ops.GraphKeys.METRIC_VARIABLES:
                from_proto = ops.get_from_proto_function(ops.GraphKeys.GLOBAL_VARIABLES)
            if from_proto and kind == 'bytes_list':
                proto_type = ops.get_collection_proto_type(key)
                if key in ops.GraphKeys._VARIABLE_COLLECTIONS:
                    for value in col_def.bytes_list.value:
                        variable = variable_objects.get(value, None)
                        if variable is None:
                            proto = proto_type()
                            proto.ParseFromString(value)
                            if not variables_have_trainable:
                                proto.trainable = key == ops.GraphKeys.TRAINABLE_VARIABLES
                            variable = from_proto(proto, import_scope=scope_to_prepend_to_names)
                            variable_objects[value] = variable
                        graph.add_to_collection(key, variable)
                else:
                    for value in col_def.bytes_list.value:
                        proto = proto_type()
                        proto.ParseFromString(value)
                        graph.add_to_collection(key, from_proto(proto, import_scope=scope_to_prepend_to_names))
            else:
                field = getattr(col_def, kind)
                if key in _COMPAT_COLLECTION_LIST:
                    logging.warning("The saved meta_graph is possibly from an older release:\n'%s' collection should be of type 'byte_list', but instead is of type '%s'.", key, kind)
                if kind == 'node_list':
                    for value in field.value:
                        col_op = graph.as_graph_element(ops.prepend_name_scope(value, scope_to_prepend_to_names))
                        graph.add_to_collection(key, col_op)
                elif kind == 'int64_list':
                    for value in field.value:
                        graph.add_to_collection(key, int(value))
                else:
                    for value in field.value:
                        graph.add_to_collection(key, ops.prepend_name_scope(value, scope_to_prepend_to_names))
        var_list = {}
        variables = graph.get_collection(ops.GraphKeys.GLOBAL_VARIABLES, scope=scope_to_prepend_to_names)
        for v in variables:
            var_list[ops.strip_name_scope(v.name, scope_to_prepend_to_names)] = v
    return (var_list, imported_return_elements)