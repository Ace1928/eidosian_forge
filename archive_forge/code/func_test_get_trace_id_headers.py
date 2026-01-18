from unittest import mock
from webob import response as webob_response
from osprofiler import _utils as utils
from osprofiler import profiler
from osprofiler.tests import test
from osprofiler import web
def test_get_trace_id_headers(self):
    profiler.init('key', base_id='y', parent_id='z')
    headers = web.get_trace_id_headers()
    self.assertEqual(sorted(headers.keys()), sorted(['X-Trace-Info', 'X-Trace-HMAC']))
    trace_info = utils.signed_unpack(headers['X-Trace-Info'], headers['X-Trace-HMAC'], ['key'])
    self.assertIn('hmac_key', trace_info)
    self.assertEqual('key', trace_info.pop('hmac_key'))
    self.assertEqual({'parent_id': 'z', 'base_id': 'y'}, trace_info)