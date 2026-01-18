import numbers as _numbers
import numpy as _np
import six as _six
import codecs
from tensorflow.python.util.tf_export import tf_export
def as_text(bytes_or_text, encoding='utf-8'):
    """Converts any string-like python input types to unicode.

  Returns the input as a unicode string. Uses utf-8 encoding for text
  by default.

  Args:
    bytes_or_text: A `bytes`, `str`, or `unicode` object.
    encoding: A string indicating the charset for decoding unicode.

  Returns:
    A `unicode` (Python 2) or `str` (Python 3) object.

  Raises:
    TypeError: If `bytes_or_text` is not a binary or unicode string.
  """
    encoding = codecs.lookup(encoding).name
    if isinstance(bytes_or_text, _six.text_type):
        return bytes_or_text
    elif isinstance(bytes_or_text, bytes):
        return bytes_or_text.decode(encoding)
    else:
        raise TypeError('Expected binary or unicode string, got %r' % bytes_or_text)