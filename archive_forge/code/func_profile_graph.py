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
def profile_graph(self, options):
    """Profile the statistics of graph nodes, organized by dataflow graph.

    Args:
      options: A dict of options. See core/profiler/g3doc/options.md.

    Returns:
      a GraphNodeProto that records the results.
    """
    opts = _build_options(options)
    tfprof_node = tfprof_output_pb2.GraphNodeProto()
    try:
        tfprof_node.ParseFromString(print_mdl.Profile('graph'.encode('utf-8'), opts.SerializeToString()))
    except message.DecodeError as e:
        sys.stderr.write('Cannot parse returned proto: %s.\n' % e)
    return tfprof_node