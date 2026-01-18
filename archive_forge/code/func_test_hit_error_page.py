import json
from webtest import TestApp
import pecan
from pecan.middleware.errordocument import ErrorDocumentMiddleware
from pecan.middleware.recursive import RecursiveMiddleware
from pecan.tests import PecanTestCase
def test_hit_error_page(self):
    r = self.app.get('/error/404')
    assert r.status_int == 200
    assert r.body == b'Error: 404'