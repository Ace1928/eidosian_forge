import abc
import re
import threading
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.framework import errors
from tensorflow.python.framework import stack
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import monitored_session
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
def on_run_start(self, request):
    """See doc of BaseDebugWrapperSession.on_run_start."""
    debug_urls, watch_opts = self._prepare_run_watch_config(request.fetches, request.feed_dict)
    return OnRunStartResponse(OnRunStartAction.DEBUG_RUN, debug_urls, debug_ops=watch_opts.debug_ops, node_name_regex_allowlist=watch_opts.node_name_regex_allowlist, op_type_regex_allowlist=watch_opts.op_type_regex_allowlist, tensor_dtype_regex_allowlist=watch_opts.tensor_dtype_regex_allowlist, tolerate_debug_op_creation_failures=watch_opts.tolerate_debug_op_creation_failures)