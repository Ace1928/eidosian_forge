from tests.compat import mock
from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from tests.unit import MockServiceWithConfigTestCase
from boto.connection import AWSAuthConnection
from boto.s3.connection import S3Connection, HostRequiredError
from boto.s3.connection import S3ResponseError, Bucket
def test_historical_behavior(self):
    self.assertEqual(self.service_connection._required_auth_capability(), ['hmac-v4-s3'])
    self.assertEqual(self.service_connection.host, 's3.amazonaws.com')