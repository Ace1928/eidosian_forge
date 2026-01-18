from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@staticmethod
def from_locations(start, end):
    return conf.lib.clang_getRange(start, end)