from tests.compat import mock
from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from tests.unit import MockServiceWithConfigTestCase
from boto.connection import AWSAuthConnection
from boto.s3.connection import S3Connection, HostRequiredError
from boto.s3.connection import S3ResponseError, Bucket
def test_head_bucket_other(self):
    self.set_http_response(status_code=405)
    with self.assertRaises(S3ResponseError) as cm:
        self.service_connection.head_bucket('you-broke-it')
    err = cm.exception
    self.assertEqual(err.status, 405)
    self.assertEqual(err.error_code, None)
    self.assertEqual(err.message, '')