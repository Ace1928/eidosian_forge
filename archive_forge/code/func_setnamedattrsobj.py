import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
@expose(NamedAttrsObject)
@validate(NamedAttrsObject)
def setnamedattrsobj(self, value):
    print(value)
    self.assertEqual(type(value), NamedAttrsObject)
    self.assertEqual(value.attr_1, 10)
    self.assertEqual(value.attr_2, 20)
    return value