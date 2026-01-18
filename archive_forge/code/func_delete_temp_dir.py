import atexit
import os
import sys
import tempfile
from absl import app
from absl.testing.absltest import *
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def delete_temp_dir(dirname=temp_dir):
    try:
        file_io.delete_recursively(dirname)
    except errors.OpError as e:
        logging.error('Error removing %s: %s', dirname, e)