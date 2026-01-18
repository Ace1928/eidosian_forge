import numbers as _numbers
import numpy as _np
import six as _six
import codecs
from tensorflow.python.util.tf_export import tf_export
@tf_export('compat.as_str_any')
def as_str_any(value, encoding='utf-8'):
    """Converts input to `str` type.

     Uses `str(value)`, except for `bytes` typed inputs, which are converted
     using `as_str`.

  Args:
    value: A object that can be converted to `str`.
    encoding: Encoding for `bytes` typed inputs.

  Returns:
    A `str` object.
  """
    if isinstance(value, bytes):
        return as_str(value, encoding=encoding)
    else:
        return str(value)