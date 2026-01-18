import os
import sys
from tensorflow.core.profiler import tfprof_log_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.profiler.internal import flops_registry  # pylint: disable=unused-import
from tensorflow.python.util.tf_export import tf_export
def merge_default_with_oplog(graph, op_log=None, run_meta=None, add_trace=True, add_trainable_var=True):
    """Merge the tfprof default extra info with caller's op_log.

  Args:
    graph: tf.Graph. If None and eager execution is not enabled, use
        default graph.
    op_log: OpLogProto proto.
    run_meta: RunMetadata proto used to complete shape information.
    add_trace: Whether to add op trace information.
    add_trainable_var: Whether to assign tf.compat.v1.trainable_variables() op
      type '_trainable_variables'.
  Returns:
    tmp_op_log: Merged OpLogProto proto.
  """
    if not graph and (not context.executing_eagerly()):
        graph = ops.get_default_graph()
    tmp_op_log = tfprof_log_pb2.OpLogProto()
    if not graph:
        return tmp_op_log
    logged_ops, string_to_id = _get_logged_ops(graph, run_meta, add_trace=add_trace, add_trainable_var=add_trainable_var)
    if not op_log:
        tmp_op_log.log_entries.extend(logged_ops.values())
    else:
        all_ops = {}
        for entry in op_log.log_entries:
            all_ops[entry.name] = entry
        for op_name, entry in logged_ops.items():
            if op_name in all_ops:
                all_ops[op_name].types.extend(entry.types)
                if entry.float_ops > 0 and all_ops[op_name].float_ops == 0:
                    all_ops[op_name].float_ops = entry.float_ops
                if entry.code_def.traces and (not all_ops[op_name].code_def.traces):
                    all_ops[op_name].code_def.MergeFrom(entry.code_def)
            else:
                all_ops[op_name] = entry
        tmp_op_log.log_entries.extend(all_ops.values())
    for s, i in string_to_id.items():
        tmp_op_log.id_to_string[i] = s
    return tmp_op_log