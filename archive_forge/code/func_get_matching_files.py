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
@tf_export(v1=['gfile.Glob'])
def get_matching_files(filename):
    """Returns a list of files that match the given pattern(s).

  Args:
    filename: string or iterable of strings. The glob pattern(s).

  Returns:
    A list of strings containing filenames that match the given pattern(s).

  Raises:
  *  errors.OpError: If there are filesystem / directory listing errors.
  *  errors.NotFoundError: If pattern to be matched is an invalid directory.
  """
    return get_matching_files_v2(filename)