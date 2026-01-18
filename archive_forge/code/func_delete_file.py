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
@tf_export(v1=['gfile.Remove'])
def delete_file(filename):
    """Deletes the file located at 'filename'.

  Args:
    filename: string, a filename

  Raises:
    errors.OpError: Propagates any errors reported by the FileSystem API.  E.g.,
    `NotFoundError` if the file does not exist.
  """
    delete_file_v2(filename)