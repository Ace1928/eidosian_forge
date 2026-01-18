from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import collections.abc
import os
@property
def is_in_system_header(self):
    """Returns true if the given source location is in a system header."""
    return conf.lib.clang_Location_isInSystemHeader(self)