from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@property
def availability(self):
    res = conf.lib.clang_getCompletionAvailability(self.obj)
    return availabilityKinds[res]