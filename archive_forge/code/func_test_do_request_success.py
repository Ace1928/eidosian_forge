import abc
from oslo_utils import uuidutils
import osprofiler.profiler
import osprofiler.web
from requests_mock.contrib import fixture as mock_fixture
import testtools
from neutronclient import client
from neutronclient.common import exceptions
def test_do_request_success(self):
    text = 'test content'
    self.requests.register_uri(METHOD, END_URL + URL, text=text)
    resp, resp_text = self.http.do_request(URL, METHOD)
    self.assertEqual(200, resp.status_code)
    self.assertEqual(text, resp_text)