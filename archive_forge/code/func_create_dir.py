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
@tf_export(v1=['gfile.MkDir'])
def create_dir(dirname):
    """Creates a directory with the name `dirname`.

  Args:
    dirname: string, name of the directory to be created

  Notes: The parent directories need to exist. Use `tf.io.gfile.makedirs`
    instead if there is the possibility that the parent dirs don't exist.

  Raises:
    errors.OpError: If the operation fails.
  """
    create_dir_v2(dirname)