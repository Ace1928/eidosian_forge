import collections
import os
import time
from tensorflow.python.keras.saving.utils_v1 import export_output as export_output_lib
from tensorflow.python.keras.saving.utils_v1 import mode_keys
from tensorflow.python.keras.saving.utils_v1 import unexported_constants
from tensorflow.python.keras.saving.utils_v1.mode_keys import KerasModeKeys as ModeKeys
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.util import compat
def get_temp_export_dir(timestamped_export_dir):
    """Builds a directory name based on the argument but starting with 'temp-'.

  This relies on the fact that TensorFlow Serving ignores subdirectories of
  the base directory that can't be parsed as integers.

  Args:
    timestamped_export_dir: the name of the eventual export directory, e.g.
      /foo/bar/<timestamp>

  Returns:
    A sister directory prefixed with 'temp-', e.g. /foo/bar/temp-<timestamp>.
  """
    dirname, basename = os.path.split(timestamped_export_dir)
    if isinstance(basename, bytes):
        str_name = basename.decode('utf-8')
    else:
        str_name = str(basename)
    temp_export_dir = os.path.join(compat.as_bytes(dirname), compat.as_bytes('temp-{}'.format(str_name)))
    return temp_export_dir