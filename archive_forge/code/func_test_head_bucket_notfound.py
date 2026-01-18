from tests.compat import mock
from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from tests.unit import MockServiceWithConfigTestCase
from boto.connection import AWSAuthConnection
from boto.s3.connection import S3Connection, HostRequiredError
from boto.s3.connection import S3ResponseError, Bucket
def test_head_bucket_notfound(self):
    self.set_http_response(status_code=404)
    with self.assertRaises(S3ResponseError) as cm:
        self.service_connection.head_bucket('totally-doesnt-exist')
    err = cm.exception
    self.assertEqual(err.status, 404)
    self.assertEqual(err.error_code, 'NoSuchBucket')
    self.assertEqual(err.message, 'The specified bucket does not exist')