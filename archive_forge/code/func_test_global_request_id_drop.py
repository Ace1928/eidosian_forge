import uuid
from oslotest import base as test_base
from testtools import matchers
import webob
import webob.dec
from oslo_middleware import request_id
def test_global_request_id_drop(self):
    """Test that bad format ids are dropped.

        This ensures that badly formatted ids are dropped entirely.
        """

    @webob.dec.wsgify
    def application(req):
        return req.environ.get(request_id.GLOBAL_REQ_ID)
    global_req = 'req-%s-bad' % uuid.uuid4()
    app = request_id.RequestId(application)
    req = webob.Request.blank('/test', headers={'X-OpenStack-Request-ID': global_req})
    res = req.get_response(app)
    res_req_id = res.headers.get(request_id.HTTP_RESP_HEADER_REQUEST_ID)
    if isinstance(res_req_id, bytes):
        res_req_id = res_req_id.decode('utf-8')
    self.assertEqual(res.body.decode('utf-8'), '')