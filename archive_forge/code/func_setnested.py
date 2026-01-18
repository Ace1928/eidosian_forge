import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
@expose(NestedOuter)
@validate(NestedOuter)
def setnested(self, value):
    print(repr(value))
    self.assertEqual(type(value), NestedOuter)
    return value