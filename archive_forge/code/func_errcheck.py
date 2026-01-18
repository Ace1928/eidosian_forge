import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
def errcheck(result, func, args):
    retval = result.value
    dll.my_free(result)
    return retval