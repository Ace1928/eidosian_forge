import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def executions(self, digest=False, begin=None, end=None):
    """Get `Execution`s or `ExecutionDigest`s this reader has read so far.

    Args:
      digest: Whether the results are returned in a digest form, i.e.,
        `ExecutionDigest` format, instead of the more detailed `Execution`
        format.
      begin: Optional beginning index for the requested execution data objects
        or their digests. Python-style negative indices are supported.
      end: Optional ending index for the requested execution data objects or
        their digests. Python-style negative indices are supported.

    Returns:
      If `digest`: a `list` of `ExecutionDigest` objects.
      Else: a `list` of `Execution` objects.
    """
    digests = self._execution_digests
    if begin is not None or end is not None:
        begin = begin or 0
        end = end or len(digests)
        digests = digests[begin:end]
    if digest:
        return digests
    else:
        return [self.read_execution(digest) for digest in digests]