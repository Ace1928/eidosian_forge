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
def create_proxy_api_files(output_files, proxy_module_root, output_dir):
    """Creates __init__.py files in proxy format for the Python API.

  Args:
    output_files: List of __init__.py file paths to create.
    proxy_module_root: Module root for proxy-import format. If specified, proxy
      files with content like `from proxy_module_root.proxy_module import *`
      will be created to enable import resolution under TensorFlow.
    output_dir: output API root directory.
  """
    for file_path in output_files:
        module = get_module(os.path.dirname(file_path), output_dir)
        if not os.path.isdir(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        contents = f'from {proxy_module_root}.{module} import *'
        with open(file_path, 'w') as fp:
            fp.write(contents)