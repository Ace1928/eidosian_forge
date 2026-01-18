from unittest import mock
import fixtures
from oslotest import base as test_base
import webob.dec
import webob.exc
from oslo_middleware import catch_errors
def test_internal_server_error(self):

    @webob.dec.wsgify
    def application(req):
        raise Exception()
    with mock.patch.object(catch_errors.LOG, 'exception') as log_exc:
        self._test_has_request_id(application, webob.exc.HTTPInternalServerError.code)
        self.assertEqual(1, log_exc.call_count)
        req_log = log_exc.call_args[0][1]
        self.assertIn('X-Auth-Token: *****', str(req_log))