from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
def multiplier_with_docstring(num, rate=2):
    """Multiplies num by rate.

  Args:
    num (int): the num you want to multiply
    rate (int): the rate for multiplication
  Returns:
    Multiplication of num by rate
  """
    return num * rate