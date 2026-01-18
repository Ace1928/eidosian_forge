from http import client as http_client
import json
import time
from unittest import mock
from keystoneauth1 import exceptions as kexc
from ironicclient.common import filecache
from ironicclient.common import http
from ironicclient import exc
from ironicclient.tests.unit import utils
def test__generic_parse_version_headers_missing_headers(self):
    response = {}
    expected = (None, None)
    result = self.test_object._generic_parse_version_headers(response.get)
    self.assertEqual(expected, result)