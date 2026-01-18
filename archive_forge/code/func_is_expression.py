from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def is_expression(self):
    """Test if this is an expression kind."""
    return conf.lib.clang_isExpression(self)