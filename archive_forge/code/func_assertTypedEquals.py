import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
def assertTypedEquals(self, a, b, convert):
    if isinstance(a, str):
        a = convert(a)
    if isinstance(b, str):
        b = convert(b)
    self.assertEqual(a, b)