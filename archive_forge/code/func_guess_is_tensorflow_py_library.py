import collections
import os
import re
import zipfile
from absl import app
import numpy as np
from tensorflow.python.debug.lib import profiling
def guess_is_tensorflow_py_library(py_file_path):
    """Guess whether a Python source file is a part of the tensorflow library.

  Special cases:
    1) Returns False for unit-test files in the library (*_test.py),
    2) Returns False for files under python/debug/examples.

  Args:
    py_file_path: full path of the Python source file in question.

  Returns:
    (`bool`) Whether the file is inferred to be a part of the tensorflow
      library.
  """
    if not is_extension_uncompiled_python_source(py_file_path) and (not is_extension_compiled_python_source(py_file_path)):
        return False
    py_file_path = _norm_abs_path(py_file_path)
    return (py_file_path.startswith(_TENSORFLOW_BASEDIR) or py_file_path.startswith(_ABSL_BASEDIR)) and (not py_file_path.endswith('_test.py')) and (os.path.normpath('tensorflow/python/debug/examples') not in os.path.normpath(py_file_path))