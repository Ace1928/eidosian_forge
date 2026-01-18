import json
from webtest import TestApp
import pecan
from pecan.middleware.errordocument import ErrorDocumentMiddleware
from pecan.middleware.recursive import RecursiveMiddleware
from pecan.tests import PecanTestCase
def test_error_with_recursion_loop(self):
    app = TestApp(RecursiveMiddleware(ErrorDocumentMiddleware(four_oh_four_app, {404: '/'})))
    r = app.get('/', expect_errors=True)
    assert r.status_int == 404
    assert r.body == b'Error: 404 Not Found.  (Error page could not be fetched)'