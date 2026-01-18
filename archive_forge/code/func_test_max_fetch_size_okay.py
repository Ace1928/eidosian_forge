import io
import urllib.error
import urllib.request
from oslo_config import cfg
import requests
from requests import exceptions
from heat.common import urlfetch
from heat.tests import common
def test_max_fetch_size_okay(self):
    url = 'http://example.com/template'
    data = b'{ "foo": "bar" }'
    response = Response(data)
    cfg.CONF.set_override('max_template_size', 500)
    mock_get = self.patchobject(requests, 'get')
    mock_get.return_value = response
    urlfetch.get(url)
    mock_get.assert_called_once_with(url, stream=True)