import json
from webtest import TestApp
import pecan
from pecan.middleware.errordocument import ErrorDocumentMiddleware
from pecan.middleware.recursive import RecursiveMiddleware
from pecan.tests import PecanTestCase
def test_error_endpoint_with_query_string(self):
    app = TestApp(RecursiveMiddleware(ErrorDocumentMiddleware(four_oh_four_app, {404: '/error/404?foo=bar'})))
    r = app.get('/', expect_errors=True)
    assert r.status_int == 404
    assert r.body == b'Error: 404\nQS: foo=bar'