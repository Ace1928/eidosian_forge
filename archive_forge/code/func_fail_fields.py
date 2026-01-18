from ctypes import *
from ctypes.test import need_symbol
from test import support
import unittest
import os
import _ctypes_test
def fail_fields(self, *fields):
    return self.get_except(type(Structure), 'X', (), {'_fields_': fields})