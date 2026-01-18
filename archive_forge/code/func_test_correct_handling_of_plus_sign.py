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
def test_correct_handling_of_plus_sign(self):
    request = HTTPRequest('GET', 'https', 's3-us-west-2.amazonaws.com', 443, 'hello+world.txt', None, {}, {}, '')
    canonical_uri = self.auth.canonical_uri(request)
    self.assertEqual(canonical_uri, 'hello%2Bworld.txt')
    request = HTTPRequest('GET', 'https', 's3-us-west-2.amazonaws.com', 443, 'hello%2Bworld.txt', None, {}, {}, '')
    canonical_uri = self.auth.canonical_uri(request)
    self.assertEqual(canonical_uri, 'hello%2Bworld.txt')