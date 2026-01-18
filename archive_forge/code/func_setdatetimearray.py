import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
@expose([datetime.datetime])
@validate([datetime.datetime])
def setdatetimearray(self, value):
    print(repr(value))
    self.assertEqual(type(value), list)
    self.assertEqual(type(value[0]), datetime.datetime)
    return value