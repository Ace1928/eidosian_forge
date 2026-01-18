import io
from tests.compat import mock, unittest
from tests.unit import AWSMockServiceTestCase
from boto.compat import StringIO
from boto.exception import BotoServerError
from boto.exception import ResumableDownloadException
from boto.exception import ResumableTransferDisposition
from boto.s3.connection import S3Connection
from boto.s3.bucket import Bucket
from boto.s3.key import Key
@mock.patch('time.sleep')
def test_502_bad_gateway(self, sleep_mock):
    weird_timeout_body = '<Error><Code>BadGateway</Code></Error>'
    self.set_http_response(status_code=502, body=weird_timeout_body)
    b = Bucket(self.service_connection, 'mybucket')
    k = b.new_key('test_failure')
    fail_file = StringIO('This will pretend to be chunk-able.')
    k.should_retry = counter(k.should_retry)
    self.assertEqual(k.should_retry.count, 0)
    with self.assertRaises(BotoServerError):
        k.send_file(fail_file)
    self.assertTrue(k.should_retry.count, 1)