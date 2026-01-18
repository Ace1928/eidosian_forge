from tests.compat import mock
from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from tests.unit import MockServiceWithConfigTestCase
from boto.connection import AWSAuthConnection
from boto.s3.connection import S3Connection, HostRequiredError
from boto.s3.connection import S3ResponseError, Bucket
def test_head_bucket_success(self):
    self.set_http_response(status_code=200)
    buck = self.service_connection.head_bucket('my-test-bucket')
    self.assertTrue(isinstance(buck, Bucket))
    self.assertEqual(buck.name, 'my-test-bucket')