import io
import urllib.error
import urllib.request
from oslo_config import cfg
import requests
from requests import exceptions
from heat.common import urlfetch
from heat.tests import common
def test_garbage(self):
    self.assertRaises(urlfetch.URLFetchError, urlfetch.get, 'wibble')