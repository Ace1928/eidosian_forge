import sys
from google.protobuf import message
from tensorflow.core.profiler import tfprof_options_pb2
from tensorflow.core.profiler import tfprof_output_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.profiler import option_builder
from tensorflow.python.profiler import tfprof_logger
from tensorflow.python.util import _pywrap_tfprof as print_mdl
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['profiler.profile'])
def profile(graph=None, run_meta=None, op_log=None, cmd='scope', options=_DEFAULT_PROFILE_OPTIONS):
    """Profile model.

    Tutorials and examples can be found in:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/python_api.md

  Args:
    graph: tf.Graph. If None and eager execution is not enabled, use default
      graph.
    run_meta: optional tensorflow.RunMetadata proto. It is necessary to
      support run time information profiling, such as time and memory.
    op_log: tensorflow.tfprof.OpLogProto proto. User can assign "types" to graph
      nodes with op_log. "types" allow user to flexibly group and account
      profiles using options['accounted_type_regexes'].
    cmd: string. Either 'op', 'scope', 'graph' or 'code'. 'op' view organizes
      profile using operation type. (e.g. MatMul) 'scope' view organizes profile
      using graph node name scope. 'graph' view organizes profile using graph
      node inputs/outputs. 'code' view organizes profile using Python call
      stack.
    options: A dict of options. See core/profiler/g3doc/options.md.

  Returns:
    If cmd is 'scope' or 'graph', returns GraphNodeProto proto.
    If cmd is 'op' or 'code', returns MultiGraphNodeProto proto.
    Side effect: stdout/file/timeline.json depending on options['output']
  """
    if not graph and (not context.executing_eagerly()):
        graph = ops.get_default_graph()
    if options == _DEFAULT_PROFILE_OPTIONS:
        options = option_builder.ProfileOptionBuilder.trainable_variables_parameter()
    op_log = tfprof_logger.merge_default_with_oplog(graph, op_log, run_meta, add_trace=cmd == 'code')
    opts = _build_options(options)
    run_meta_str = run_meta.SerializeToString() if run_meta else b''
    graph_str = _graph_string(graph)
    if cmd == 'code' or cmd == 'op':
        tfprof_node = tfprof_output_pb2.MultiGraphNodeProto()
        ret = print_mdl.PrintModelAnalysis(graph_str, run_meta_str, op_log.SerializeToString(), cmd.encode('utf-8'), opts.SerializeToString())
        try:
            tfprof_node.ParseFromString(ret)
        except message.DecodeError as e:
            sys.stderr.write('Cannot parse returned proto: %s.\n' % e)
    elif cmd == 'graph' or cmd == 'scope':
        tfprof_node = tfprof_output_pb2.GraphNodeProto()
        ret = print_mdl.PrintModelAnalysis(graph_str, run_meta_str, op_log.SerializeToString(), cmd.encode('utf-8'), opts.SerializeToString())
        try:
            tfprof_node.ParseFromString(ret)
        except message.DecodeError as e:
            sys.stderr.write('Cannot parse returned proto: %s.\n' % e)
    else:
        raise errors.InvalidArgumentError(None, None, 'unknown cmd: %s\n' % cmd)
    return tfprof_node