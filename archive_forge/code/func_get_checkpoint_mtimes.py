import collections
import os.path
import re
import time
from google.protobuf import text_format
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import training_util
from tensorflow.python.training.checkpoint_state_pb2 import CheckpointState
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@deprecation.deprecated(date=None, instructions='Use standard file utilities to get mtimes.')
@tf_export(v1=['train.get_checkpoint_mtimes'])
def get_checkpoint_mtimes(checkpoint_prefixes):
    """Returns the mtimes (modification timestamps) of the checkpoints.

  Globs for the checkpoints pointed to by `checkpoint_prefixes`.  If the files
  exist, collect their mtime.  Both V2 and V1 checkpoints are considered, in
  that priority.

  This is the recommended way to get the mtimes, since it takes into account
  the naming difference between V1 and V2 formats.

  Note: If not all checkpoints exist, the length of the returned mtimes list
  will be smaller than the length of `checkpoint_prefixes` list, so mapping
  checkpoints to corresponding mtimes will not be possible.

  Args:
    checkpoint_prefixes: a list of checkpoint paths, typically the results of
      `Saver.save()` or those of `tf.train.latest_checkpoint()`, regardless of
      sharded/non-sharded or V1/V2.
  Returns:
    A list of mtimes (in microseconds) of the found checkpoints.
  """
    mtimes = []

    def match_maybe_append(pathname):
        fnames = file_io.get_matching_files(pathname)
        if fnames:
            mtimes.append(file_io.stat(fnames[0]).mtime_nsec / 1000000000.0)
            return True
        return False
    for checkpoint_prefix in checkpoint_prefixes:
        pathname = _prefix_to_checkpoint_path(checkpoint_prefix, saver_pb2.SaverDef.V2)
        if match_maybe_append(pathname):
            continue
        match_maybe_append(checkpoint_prefix)
    return mtimes