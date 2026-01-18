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
@tf_export('io.gfile.listdir')
def list_directory_v2(path):
    """Returns a list of entries contained within a directory.

  The list is in arbitrary order. It does not contain the special entries "."
  and "..".

  Args:
    path: string, path to a directory

  Returns:
    [filename1, filename2, ... filenameN] as strings

  Raises:
    errors.NotFoundError if directory doesn't exist
  """
    if not is_directory(path):
        raise errors.NotFoundError(node_def=None, op=None, message='Could not find directory {}'.format(path))
    return [compat.as_str_any(filename) for filename in _pywrap_file_io.GetChildren(compat.path_to_bytes(path))]