import io
import urllib.error
import urllib.request
from oslo_config import cfg
import requests
from requests import exceptions
from heat.common import urlfetch
from heat.tests import common
def test_max_fetch_size_error(self):
    url = 'http://example.com/template'
    data = b'{ "foo": "bar" }'
    response = Response(data)
    cfg.CONF.set_override('max_template_size', 5)
    mock_get = self.patchobject(requests, 'get')
    mock_get.return_value = response
    exception = self.assertRaises(urlfetch.URLFetchError, urlfetch.get, url)
    self.assertIn('Template exceeds', str(exception))
    mock_get.assert_called_once_with(url, stream=True)