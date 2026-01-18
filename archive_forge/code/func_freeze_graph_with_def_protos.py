import argparse
import re
import sys
from absl import app
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.core.protobuf.meta_graph_pb2 import MetaGraphDef
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.client import session
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import importer
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training import saver as saver_lib
def freeze_graph_with_def_protos(input_graph_def, input_saver_def, input_checkpoint, output_node_names, restore_op_name, filename_tensor_name, output_graph, clear_devices, initializer_nodes, variable_names_whitelist='', variable_names_denylist='', input_meta_graph_def=None, input_saved_model_dir=None, saved_model_tags=None, checkpoint_version=saver_pb2.SaverDef.V2):
    """Converts all variables in a graph and checkpoint into constants.

  Args:
    input_graph_def: A `GraphDef`.
    input_saver_def: A `SaverDef` (optional).
    input_checkpoint: The prefix of a V1 or V2 checkpoint, with V2 taking
      priority.  Typically the result of `Saver.save()` or that of
      `tf.train.latest_checkpoint()`, regardless of sharded/non-sharded or
      V1/V2.
    output_node_names: The name(s) of the output nodes, comma separated.
    restore_op_name: Unused.
    filename_tensor_name: Unused.
    output_graph: String where to write the frozen `GraphDef`.
    clear_devices: A Bool whether to remove device specifications.
    initializer_nodes: Comma separated string of initializer nodes to run before
                       freezing.
    variable_names_whitelist: The set of variable names to convert (optional, by
                              default, all variables are converted).
    variable_names_denylist: The set of variable names to omit converting
                              to constants (optional).
    input_meta_graph_def: A `MetaGraphDef` (optional),
    input_saved_model_dir: Path to the dir with TensorFlow 'SavedModel' file
                           and variables (optional).
    saved_model_tags: Group of comma separated tag(s) of the MetaGraphDef to
                      load, in string format (optional).
    checkpoint_version: Tensorflow variable file format (saver_pb2.SaverDef.V1
                        or saver_pb2.SaverDef.V2)

  Returns:
    Location of the output_graph_def.
  """
    del restore_op_name, filename_tensor_name
    if not input_saved_model_dir and (not checkpoint_management.checkpoint_exists(input_checkpoint)):
        raise ValueError("Input checkpoint '" + input_checkpoint + "' doesn't exist!")
    if not output_node_names:
        raise ValueError('You need to supply the name of a node to --output_node_names.')
    if clear_devices:
        if input_meta_graph_def:
            for node in input_meta_graph_def.graph_def.node:
                node.device = ''
        elif input_graph_def:
            for node in input_graph_def.node:
                node.device = ''
    if input_graph_def:
        _ = importer.import_graph_def(input_graph_def, name='')
    with session.Session() as sess:
        if input_saver_def:
            saver = saver_lib.Saver(saver_def=input_saver_def, write_version=checkpoint_version)
            saver.restore(sess, input_checkpoint)
        elif input_meta_graph_def:
            restorer = saver_lib.import_meta_graph(input_meta_graph_def, clear_devices=True)
            restorer.restore(sess, input_checkpoint)
            if initializer_nodes:
                sess.run(initializer_nodes.replace(' ', '').split(','))
        elif input_saved_model_dir:
            if saved_model_tags is None:
                saved_model_tags = []
            loader.load(sess, saved_model_tags, input_saved_model_dir)
        else:
            var_list = {}
            reader = py_checkpoint_reader.NewCheckpointReader(input_checkpoint)
            var_to_shape_map = reader.get_variable_to_shape_map()
            all_partition_variable_names = [tensor.name.split(':')[0] for op in sess.graph.get_operations() for tensor in op.values() if re.search('/part_\\d+/', tensor.name)]
            has_partition_var = False
            for key in var_to_shape_map:
                try:
                    tensor = sess.graph.get_tensor_by_name(key + ':0')
                    if any((key in name for name in all_partition_variable_names)):
                        has_partition_var = True
                except KeyError:
                    continue
                var_list[key] = tensor
            try:
                saver = saver_lib.Saver(var_list=var_list, write_version=checkpoint_version)
            except TypeError as e:
                if has_partition_var:
                    raise ValueError('Models containing partition variables cannot be converted from checkpoint files. Please pass in a SavedModel using the flag --input_saved_model_dir.')
                elif _has_no_variables(sess):
                    raise ValueError('No variables were found in this model. It is likely the model was frozen previously. You cannot freeze a graph twice.')
                    return 0
                else:
                    raise e
            saver.restore(sess, input_checkpoint)
            if initializer_nodes:
                sess.run(initializer_nodes.replace(' ', '').split(','))
        variable_names_whitelist = variable_names_whitelist.replace(' ', '').split(',') if variable_names_whitelist else None
        variable_names_denylist = variable_names_denylist.replace(' ', '').split(',') if variable_names_denylist else None
        if input_meta_graph_def:
            output_graph_def = convert_to_constants.convert_variables_to_constants(sess, input_meta_graph_def.graph_def, output_node_names.replace(' ', '').split(','), variable_names_whitelist=variable_names_whitelist, variable_names_blacklist=variable_names_denylist)
        else:
            output_graph_def = convert_to_constants.convert_variables_to_constants(sess, input_graph_def, output_node_names.replace(' ', '').split(','), variable_names_whitelist=variable_names_whitelist, variable_names_blacklist=variable_names_denylist)
    if output_graph:
        with gfile.GFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString(deterministic=True))
    return output_graph_def