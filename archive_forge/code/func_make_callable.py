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
def make_callable(self, fetches, feed_list=None, accept_options=False):
    runner = self._sess.make_callable(fetches, feed_list=feed_list, accept_options=True)

    def wrapped_runner(*runner_args, **kwargs):
        return self.run(None, feed_dict=None, options=kwargs.get('options', None), run_metadata=kwargs.get('run_metadata', None), callable_runner=runner, callable_runner_args=runner_args)
    return wrapped_runner