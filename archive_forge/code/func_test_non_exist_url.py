import io
import urllib.error
import urllib.request
from oslo_config import cfg
import requests
from requests import exceptions
from heat.common import urlfetch
from heat.tests import common
def test_non_exist_url(self):
    url = 'http://non-exist.com/template'
    mock_get = self.patchobject(requests, 'get')
    mock_get.side_effect = exceptions.Timeout()
    self.assertRaises(urlfetch.URLFetchError, urlfetch.get, url)
    mock_get.assert_called_once_with(url, stream=True)