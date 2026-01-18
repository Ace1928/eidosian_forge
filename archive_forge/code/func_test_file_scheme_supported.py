import io
import urllib.error
import urllib.request
from oslo_config import cfg
import requests
from requests import exceptions
from heat.common import urlfetch
from heat.tests import common
def test_file_scheme_supported(self):
    data = '{ "foo": "bar" }'
    url = 'file:///etc/profile'
    mock_open = self.patchobject(urllib.request, 'urlopen')
    mock_open.return_value = io.StringIO(data)
    self.assertEqual(data, urlfetch.get(url, allowed_schemes=['file']))
    mock_open.assert_called_once_with(url)