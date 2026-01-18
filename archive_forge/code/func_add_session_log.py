from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import test_util
from tensorflow.python.summary.writer import writer
from tensorflow.python.summary.writer import writer_cache
def add_session_log(self, session_log, global_step=None):
    self._added_session_logs.append(session_log)