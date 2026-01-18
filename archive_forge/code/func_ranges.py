from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@property
def ranges(self):

    class RangeIterator(object):

        def __init__(self, diag):
            self.diag = diag

        def __len__(self):
            return int(conf.lib.clang_getDiagnosticNumRanges(self.diag))

        def __getitem__(self, key):
            if key >= len(self):
                raise IndexError
            return conf.lib.clang_getDiagnosticRange(self.diag, key)
    return RangeIterator(self)