import uuid
from oslotest import base as test_base
from testtools import matchers
import webob
import webob.dec
from oslo_middleware import request_id
def test_compat_headers(self):
    """Test that compat headers are set

        Compat headers might exist on a super class to support
        previous API contracts. This ensures that you can set that to
        a list of headers and those values are the same as the
        request_id.

        """

    @webob.dec.wsgify
    def application(req):
        return req.environ[request_id.ENV_REQUEST_ID]
    app = AltHeader(application)
    req = webob.Request.blank('/test')
    res = req.get_response(app)
    res_req_id = res.headers.get(request_id.HTTP_RESP_HEADER_REQUEST_ID)
    self.assertEqual(res.headers.get('x-compute-req-id'), res_req_id)
    self.assertEqual(res.headers.get('x-silly-id'), res_req_id)