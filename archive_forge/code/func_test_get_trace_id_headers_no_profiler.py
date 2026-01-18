from unittest import mock
from webob import response as webob_response
from osprofiler import _utils as utils
from osprofiler import profiler
from osprofiler.tests import test
from osprofiler import web
@mock.patch('osprofiler.profiler.get')
def test_get_trace_id_headers_no_profiler(self, mock_get_profiler):
    mock_get_profiler.return_value = False
    headers = web.get_trace_id_headers()
    self.assertEqual(headers, {})