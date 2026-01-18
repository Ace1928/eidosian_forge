from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def is_preprocessing(self):
    """Test if this is a preprocessing kind."""
    return conf.lib.clang_isPreprocessing(self)