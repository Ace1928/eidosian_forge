from __future__ import absolute_import
import sys
import types
def iterbytes(s):
    return (ord(c) for c in s)