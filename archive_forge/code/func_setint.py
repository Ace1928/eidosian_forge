import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
@expose(int)
@validate(int)
def setint(self, value):
    print(repr(value))
    self.assertEqual(type(value), int)
    return value