from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import re
import struct
import sys
import textwrap
import six
from six.moves import range  # pylint: disable=redefined-builtin
def get_module_object_and_name(globals_dict):
    """Returns the module that defines a global environment, and its name.

  Args:
    globals_dict: A dictionary that should correspond to an environment
      providing the values of the globals.

  Returns:
    _ModuleObjectAndName - pair of module object & module name.
    Returns (None, None) if the module could not be identified.
  """
    name = globals_dict.get('__name__', None)
    module = sys.modules.get(name, None)
    return _ModuleObjectAndName(module, sys.argv[0] if name == '__main__' else name)