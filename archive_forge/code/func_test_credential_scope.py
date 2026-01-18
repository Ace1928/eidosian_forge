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
def test_credential_scope(self):
    auth = HmacAuthV4Handler('iam.amazonaws.com', mock.Mock(), self.provider)
    request = HTTPRequest('POST', 'https', 'iam.amazonaws.com', 443, '/', '/', {'Action': 'ListAccountAliases', 'Version': '2010-05-08'}, {'Content-Length': '44', 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8', 'X-Amz-Date': '20130808T013210Z'}, 'Action=ListAccountAliases&Version=2010-05-08')
    credential_scope = auth.credential_scope(request)
    region_name = credential_scope.split('/')[1]
    self.assertEqual(region_name, 'us-east-1')
    auth = HmacAuthV4Handler('iam.us-gov.amazonaws.com', mock.Mock(), self.provider)
    request = HTTPRequest('POST', 'https', 'iam.us-gov.amazonaws.com', 443, '/', '/', {'Action': 'ListAccountAliases', 'Version': '2010-05-08'}, {'Content-Length': '44', 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8', 'X-Amz-Date': '20130808T013210Z'}, 'Action=ListAccountAliases&Version=2010-05-08')
    credential_scope = auth.credential_scope(request)
    region_name = credential_scope.split('/')[1]
    self.assertEqual(region_name, 'us-gov-west-1')
    auth = HmacAuthV4Handler('iam.us-west-1.amazonaws.com', mock.Mock(), self.provider)
    request = HTTPRequest('POST', 'https', 'iam.us-west-1.amazonaws.com', 443, '/', '/', {'Action': 'ListAccountAliases', 'Version': '2010-05-08'}, {'Content-Length': '44', 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8', 'X-Amz-Date': '20130808T013210Z'}, 'Action=ListAccountAliases&Version=2010-05-08')
    credential_scope = auth.credential_scope(request)
    region_name = credential_scope.split('/')[1]
    self.assertEqual(region_name, 'us-west-1')
    auth = HmacAuthV4Handler('localhost', mock.Mock(), self.provider, service_name='iam')
    request = HTTPRequest('POST', 'http', 'localhost', 8080, '/', '/', {'Action': 'ListAccountAliases', 'Version': '2010-05-08'}, {'Content-Length': '44', 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8', 'X-Amz-Date': '20130808T013210Z'}, 'Action=ListAccountAliases&Version=2010-05-08')
    credential_scope = auth.credential_scope(request)
    timestamp, region, service, v = credential_scope.split('/')
    self.assertEqual(region, 'localhost')
    self.assertEqual(service, 'iam')