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
def getobjectarrayattribute(self):
    obj = NestedOuter()
    obj.inner_array = [NestedInner(12), NestedInner(13)]
    return obj