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
def add_step(self, step, run_meta):
    """Add statistics of a step.

    Args:
      step: int, An id used to group one or more different `run_meta` together.
        When profiling with the profile_xxx APIs, user can use the `step` id in
        the `options` to profile these `run_meta` together.
      run_meta: RunMetadata proto that contains statistics of a session run.
    """
    op_log = tfprof_logger.merge_default_with_oplog(self._graph, run_meta=run_meta)
    self._coverage = print_mdl.AddStep(step, _graph_string(self._graph), run_meta.SerializeToString(), op_log.SerializeToString())