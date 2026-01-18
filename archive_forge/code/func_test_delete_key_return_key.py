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
def test_delete_key_return_key(self):
    self.set_http_response(status_code=204, body='')
    b = Bucket(self.service_connection, 'mybucket')
    key = b.delete_key('fookey')
    self.assertIsNotNone(key)