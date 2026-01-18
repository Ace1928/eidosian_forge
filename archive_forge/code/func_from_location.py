from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@staticmethod
def from_location(tu, location):
    cursor = conf.lib.clang_getCursor(tu, location)
    cursor._tu = tu
    return cursor