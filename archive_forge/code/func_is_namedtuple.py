import collections as _collections
import enum
import typing
from typing import Protocol
import six as _six
import wrapt as _wrapt
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util.compat import collections_abc as _collections_abc
def is_namedtuple(instance, strict=False):
    """Returns True iff `instance` is a `namedtuple`.

  Args:
    instance: An instance of a Python object.
    strict: If True, `instance` is considered to be a `namedtuple` only if it is
      a "plain" namedtuple. For instance, a class inheriting from a `namedtuple`
      will be considered to be a `namedtuple` iff `strict=False`.

  Returns:
    True if `instance` is a `namedtuple`.
  """
    return _pywrap_utils.IsNamedtuple(instance, strict)