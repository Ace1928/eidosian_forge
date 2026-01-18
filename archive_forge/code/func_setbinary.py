import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
@expose(wsme.types.binary)
@validate(wsme.types.binary)
def setbinary(self, value):
    print(repr(value))
    self.assertEqual(type(value), bytes)
    return value