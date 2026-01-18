import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
@expose(decimal.Decimal)
def getdecimal(self):
    return decimal.Decimal('3.14159265')