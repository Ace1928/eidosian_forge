from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
def fn_with_code_in_docstring():
    """This has code in the docstring.



  Example:
    x = fn_with_code_in_docstring()
    indentation_matters = True



  Returns:
    True.
  """
    return True