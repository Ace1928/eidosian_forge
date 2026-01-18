from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
def fn_with_multiple_defaults(first='first', last='last', late='late'):
    """Function with kwarg and defaults.

  :key first: Description of first.
  :key last: Description of last.
  :key late: Description of late.
  """
    del last, late
    return first