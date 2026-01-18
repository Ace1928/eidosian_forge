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
def create_meta_graph_def(meta_info_def=None, graph_def=None, saver_def=None, collection_list=None, graph=None, export_scope=None, exclude_nodes=None, clear_extraneous_savers=False, strip_default_attrs=False):
    """Construct and returns a `MetaGraphDef` protocol buffer.

  Args:
    meta_info_def: `MetaInfoDef` protocol buffer.
    graph_def: `GraphDef` protocol buffer.
    saver_def: `SaverDef` protocol buffer.
    collection_list: List of string keys to collect.
    graph: The `Graph` to create `MetaGraphDef` out of.
    export_scope: Optional `string`. Name scope to remove.
    exclude_nodes: An iterable of nodes or `string` node names to omit from all
      collection, or None.
    clear_extraneous_savers: Remove any preexisting SaverDefs from the SAVERS
        collection.  Note this method does not alter the graph, so any
        extraneous Save/Restore ops should have been removed already, as needed.
    strip_default_attrs: Boolean. If `True`, default-valued attributes will be
        removed from the NodeDefs. For a detailed guide, see
        [Stripping Default-Valued Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).

  Returns:
    MetaGraphDef protocol buffer.

  Raises:
    TypeError: If the arguments are not of the correct proto buffer type.
  """
    if graph and (not isinstance(graph, ops.Graph)):
        raise TypeError(f'graph must be of type Graph. Received type: {type(graph)}.')
    if meta_info_def and (not isinstance(meta_info_def, meta_graph_pb2.MetaGraphDef.MetaInfoDef)):
        raise TypeError(f'meta_info_def must be of type MetaInfoDef. Received type: {type(meta_info_def)}.')
    if graph_def and (not isinstance(graph_def, graph_pb2.GraphDef)):
        raise TypeError(f'graph_def must be of type GraphDef. Received type: {type(graph_def)}.')
    if saver_def and (not isinstance(saver_def, saver_pb2.SaverDef)):
        raise TypeError(f'saver_def must be of type SaverDef. Received type: {type(saver_def)}.')
    graph = graph or ops.get_default_graph()
    meta_graph_def = meta_graph_pb2.MetaGraphDef()
    if not meta_info_def:
        meta_info_def = meta_graph_pb2.MetaGraphDef.MetaInfoDef()
    meta_info_def.tensorflow_version = versions.__version__
    meta_info_def.tensorflow_git_version = versions.__git_version__
    meta_graph_def.meta_info_def.MergeFrom(meta_info_def)
    if not graph_def:
        meta_graph_def.graph_def.MergeFrom(graph.as_graph_def(add_shapes=True))
    else:
        meta_graph_def.graph_def.MergeFrom(graph_def)
    if len(meta_graph_def.meta_info_def.stripped_op_list.op) == 0:
        meta_graph_def.meta_info_def.stripped_op_list.MergeFrom(stripped_op_list_for_graph(meta_graph_def.graph_def))
    if strip_default_attrs:
        strip_graph_default_valued_attrs(meta_graph_def)
    if saver_def:
        meta_graph_def.saver_def.MergeFrom(saver_def)
    if collection_list is not None:
        clist = collection_list
    else:
        clist = graph.get_all_collection_keys()
    for ctype in clist:
        if clear_extraneous_savers and ctype == ops.GraphKeys.SAVERS:
            from_proto = ops.get_from_proto_function(ctype)
            add_collection_def(meta_graph_def, ctype, graph=graph, export_scope=export_scope, exclude_nodes=exclude_nodes, override_contents=[from_proto(saver_def)])
        else:
            add_collection_def(meta_graph_def, ctype, graph=graph, export_scope=export_scope, exclude_nodes=exclude_nodes)
    return meta_graph_def