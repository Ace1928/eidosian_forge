from datetime import datetime, date
from decimal import Decimal
from json import loads
from webtest import TestApp
from webob.multidict import MultiDict
from pecan.jsonify import jsonify, encode, ResultProxy, RowProxy
from pecan import Pecan, expose
from pecan.tests import PecanTestCase
def create_fake_proxies(self):

    class FakeSAObject(object):

        def __init__(self):
            self._sa_class_manager = object()
            self._sa_instance_state = 'awesome'
            self.id = 1
            self.first_name = 'Jonathan'
            self.last_name = 'LaCour'

    class FakeResultProxy(ResultProxy):

        def __init__(self):
            self.rowcount = -1
            self.rows = []

        def __iter__(self):
            return iter(self.rows)

        def append(self, row):
            self.rows.append(row)

    class FakeRowProxy(RowProxy):

        def __init__(self, arg=None):
            self.row = dict(arg)

        def __getitem__(self, key):
            return self.row.__getitem__(key)

        def keys(self):
            return self.row.keys()
    self.sa_object = FakeSAObject()
    self.result_proxy = FakeResultProxy()
    self.result_proxy.append(FakeRowProxy([('id', 1), ('first_name', 'Jonathan'), ('last_name', 'LaCour')]))
    self.result_proxy.append(FakeRowProxy([('id', 2), ('first_name', 'Ryan'), ('last_name', 'Petrello')]))
    self.row_proxy = FakeRowProxy([('id', 1), ('first_name', 'Jonathan'), ('last_name', 'LaCour')])