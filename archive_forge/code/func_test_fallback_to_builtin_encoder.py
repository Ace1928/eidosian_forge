from datetime import datetime, date
from decimal import Decimal
from json import loads
from webtest import TestApp
from webob.multidict import MultiDict
from pecan.jsonify import jsonify, encode, ResultProxy, RowProxy
from pecan import Pecan, expose
from pecan.tests import PecanTestCase
def test_fallback_to_builtin_encoder(self):

    class Foo(object):
        pass
    self.assertRaises(TypeError, encode, Foo())