from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
def fn_with_kwarg_and_defaults(arg1, arg2, opt=True, **kwargs):
    """Function with kwarg and defaults.

  :param arg1: Description of arg1.
  :param arg2: Description of arg2.
  :key arg3: Description of arg3.
  """
    del arg1, arg2, opt
    return kwargs.get('arg3')