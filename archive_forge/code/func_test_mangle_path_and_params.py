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
def test_mangle_path_and_params(self):
    request = HTTPRequest(method='GET', protocol='https', host='awesome-bucket.s3-us-west-2.amazonaws.com', port=443, path='/?delete&max-keys=0', auth_path=None, params={'key': 'why hello there', 'max-keys': 1}, headers={'User-Agent': 'Boto', 'X-AMZ-Content-sha256': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'X-AMZ-Date': '20130605T193245Z'}, body='')
    mod_req = self.auth.mangle_path_and_params(request)
    self.assertEqual(mod_req.path, '/?delete&max-keys=0')
    self.assertEqual(mod_req.auth_path, '/')
    self.assertEqual(mod_req.params, {'max-keys': '0', 'key': 'why hello there', 'delete': ''})