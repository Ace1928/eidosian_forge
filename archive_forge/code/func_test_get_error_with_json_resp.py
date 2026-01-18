import io
import logging
from testtools import matchers
from keystoneclient import exceptions
from keystoneclient import httpclient
from keystoneclient import session
from keystoneclient.tests.unit import utils
def test_get_error_with_json_resp(self):
    with self.deprecations.expect_deprecations_here():
        cl = get_authed_client()
    err_response = {'error': {'code': 400, 'title': 'Error title', 'message': 'Error message string'}}
    self.stub_url('GET', status_code=400, json=err_response)
    exc_raised = False
    try:
        with self.deprecations.expect_deprecations_here():
            cl.get('/hi')
    except exceptions.BadRequest as exc:
        exc_raised = True
        self.assertEqual(exc.message, 'Error message string (HTTP 400)')
    self.assertTrue(exc_raised, 'Exception not raised.')