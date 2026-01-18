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
def test_clean_region_name(self):
    cleaned = self.auth.clean_region_name('us-west-2')
    self.assertEqual(cleaned, 'us-west-2')
    cleaned = self.auth.clean_region_name('s3-us-west-2')
    self.assertEqual(cleaned, 'us-west-2')
    cleaned = self.auth.clean_region_name('s3.amazonaws.com')
    self.assertEqual(cleaned, 's3.amazonaws.com')
    cleaned = self.auth.clean_region_name('something-s3-us-west-2')
    self.assertEqual(cleaned, 'something-s3-us-west-2')