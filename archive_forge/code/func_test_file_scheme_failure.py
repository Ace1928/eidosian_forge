import io
import urllib.error
import urllib.request
from oslo_config import cfg
import requests
from requests import exceptions
from heat.common import urlfetch
from heat.tests import common
def test_file_scheme_failure(self):
    url = 'file:///etc/profile'
    mock_open = self.patchobject(urllib.request, 'urlopen')
    mock_open.side_effect = urllib.error.URLError('oops')
    self.assertRaises(urlfetch.URLFetchError, urlfetch.get, url, allowed_schemes=['file'])
    mock_open.assert_called_once_with(url)