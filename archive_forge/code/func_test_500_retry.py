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
def test_500_retry(self, sleep_mock):
    self.set_http_response(status_code=500)
    b = Bucket(self.service_connection, 'mybucket')
    k = b.new_key('test_failure')
    fail_file = StringIO('This will attempt to retry.')
    with self.assertRaises(BotoServerError):
        k.send_file(fail_file)