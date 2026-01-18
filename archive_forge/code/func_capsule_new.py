import sys
import ctypes
from ctypes import *
import unittest
def capsule_new(p):
    return PyCapsule_New(addressof(p), None, None)