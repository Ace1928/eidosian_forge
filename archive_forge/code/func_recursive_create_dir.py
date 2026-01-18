import binascii
import os
from posixpath import join as urljoin
import uuid
import six
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import _pywrap_file_io
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['gfile.MakeDirs'])
def recursive_create_dir(dirname):
    """Creates a directory and all parent/intermediate directories.

  It succeeds if dirname already exists and is writable.

  Args:
    dirname: string, name of the directory to be created

  Raises:
    errors.OpError: If the operation fails.
  """
    recursive_create_dir_v2(dirname)