from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def register_enumerations():
    for name, value in clang.enumerations.TokenKinds:
        TokenKind.register(value, name)