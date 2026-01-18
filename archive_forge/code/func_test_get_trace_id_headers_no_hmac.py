from unittest import mock
from webob import response as webob_response
from osprofiler import _utils as utils
from osprofiler import profiler
from osprofiler.tests import test
from osprofiler import web
def test_get_trace_id_headers_no_hmac(self):
    profiler.init(None, base_id='y', parent_id='z')
    headers = web.get_trace_id_headers()
    self.assertEqual(headers, {})