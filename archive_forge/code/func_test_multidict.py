from datetime import datetime, date
from decimal import Decimal
from json import loads
from webtest import TestApp
from webob.multidict import MultiDict
from pecan.jsonify import jsonify, encode, ResultProxy, RowProxy
from pecan import Pecan, expose
from pecan.tests import PecanTestCase
def test_multidict(self):
    md = MultiDict()
    md.add('arg', 'foo')
    md.add('arg', 'bar')
    result = encode(md)
    assert loads(result) == {'arg': ['foo', 'bar']}