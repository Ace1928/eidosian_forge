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
def get_timestamped_export_dir(export_dir_base):
    """Builds a path to a new subdirectory within the base directory.

  Each export is written into a new subdirectory named using the
  current time.  This guarantees monotonically increasing version
  numbers even across multiple runs of the pipeline.
  The timestamp used is the number of seconds since epoch UTC.

  Args:
    export_dir_base: A string containing a directory to write the exported
        graph and checkpoints.
  Returns:
    The full path of the new subdirectory (which is not actually created yet).

  Raises:
    RuntimeError: if repeated attempts fail to obtain a unique timestamped
      directory name.
  """
    attempts = 0
    while attempts < MAX_DIRECTORY_CREATION_ATTEMPTS:
        timestamp = int(time.time())
        result_dir = os.path.join(compat.as_bytes(export_dir_base), compat.as_bytes(str(timestamp)))
        if not gfile.Exists(result_dir):
            return result_dir
        time.sleep(1)
        attempts += 1
        logging.warning('Directory {} already exists; retrying (attempt {}/{})'.format(compat.as_str(result_dir), attempts, MAX_DIRECTORY_CREATION_ATTEMPTS))
    raise RuntimeError('Failed to obtain a unique export directory name after {} attempts.'.format(MAX_DIRECTORY_CREATION_ATTEMPTS))