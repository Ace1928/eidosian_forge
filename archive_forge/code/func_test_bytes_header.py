import copy
import pickle
import os
from tests.compat import unittest, mock
from tests.unit import MockServiceWithConfigTestCase
from nose.tools import assert_equal
from boto.auth import HmacAuthV4Handler
from boto.auth import S3HmacAuthV4Handler
from boto.auth import detect_potential_s3sigv4
from boto.auth import detect_potential_sigv4
from boto.connection import HTTPRequest
from boto.provider import Provider
from boto.regioninfo import RegionInfo
def test_bytes_header(self):
    auth = HmacAuthV4Handler('glacier.us-east-1.amazonaws.com', mock.Mock(), self.provider)
    request = HTTPRequest('GET', 'http', 'glacier.us-east-1.amazonaws.com', 80, 'x/./././x .html', None, {}, {'x-amz-glacier-version': '2012-06-01', 'x-amz-hash': b'f00'}, '')
    canonical = auth.canonical_request(request)
    self.assertIn('f00', canonical)