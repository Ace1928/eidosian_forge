from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@property
def element_count(self):
    """Retrieve the number of elements in this type.

        Returns an int.

        If the Type is not an array or vector, this raises.
        """
    result = conf.lib.clang_getNumElements(self)
    if result < 0:
        raise Exception('Type does not have elements.')
    return result