import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
@expose([wsme.types.bytes])
@validate([wsme.types.bytes])
def setbytesarray(self, value):
    print(repr(value))
    self.assertEqual(type(value), list)
    self.assertEqual(type(value[0]), wsme.types.bytes)
    return value