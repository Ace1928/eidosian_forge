from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@property
def briefComment(self):
    if conf.function_exists('clang_getCompletionBriefComment'):
        return conf.lib.clang_getCompletionBriefComment(self.obj)
    return _CXString()