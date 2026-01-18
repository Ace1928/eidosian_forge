from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@property
def category_number(self):
    """The category number for this diagnostic or 0 if unavailable."""
    return conf.lib.clang_getDiagnosticCategory(self)