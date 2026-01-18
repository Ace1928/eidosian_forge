from json import dumps
from webtest import TestApp
from pecan import Pecan, expose, abort
from pecan.tests import PecanTestCase
def test_generics_not_allowed(self):

    class C(object):

        def _default(self):
            pass

        def _lookup(self):
            pass

        def _route(self):
            pass
    for method in (C._default, C._lookup, C._route):
        self.assertRaises(ValueError, expose(generic=True), getattr(method, '__func__', method))