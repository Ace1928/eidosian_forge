import argparse
import collections
import importlib
import os
import sys
from tensorflow.python.tools.api.generator import doc_srcs
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export
import sys as _sys
from tensorflow.python.util import module_wrapper as _module_wrapper
def get_module(dir_path, relative_to_dir):
    """Get module that corresponds to path relative to relative_to_dir.

  Args:
    dir_path: Path to directory.
    relative_to_dir: Get module relative to this directory.

  Returns:
    Name of module that corresponds to the given directory.
  """
    dir_path = dir_path[len(relative_to_dir):]
    dir_path = dir_path.replace(os.sep, '/')
    return dir_path.replace('/', '.').strip('.')