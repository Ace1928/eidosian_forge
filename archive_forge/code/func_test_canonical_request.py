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
def test_canonical_request(self):
    expected = 'GET\n/\nmax-keys=0\nhost:awesome-bucket.s3-us-west-2.amazonaws.com\nuser-agent:Boto\nx-amz-content-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\nx-amz-date:20130605T193245Z\n\nhost;user-agent;x-amz-content-sha256;x-amz-date\ne3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    authed_req = self.auth.canonical_request(self.awesome_bucket_request)
    self.assertEqual(authed_req, expected)
    request = copy.copy(self.awesome_bucket_request)
    request.path = request.auth_path = '/?max-keys=0'
    request.params = {}
    expected = 'GET\n/\nmax-keys=0\nhost:awesome-bucket.s3-us-west-2.amazonaws.com\nuser-agent:Boto\nx-amz-content-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\nx-amz-date:20130605T193245Z\n\nhost;user-agent;x-amz-content-sha256;x-amz-date\ne3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    request = self.auth.mangle_path_and_params(request)
    authed_req = self.auth.canonical_request(request)
    self.assertEqual(authed_req, expected)