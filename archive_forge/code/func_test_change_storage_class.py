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
def test_change_storage_class(self):
    self.set_http_response(status_code=200)
    b = Bucket(self.service_connection, 'mybucket')
    k = b.get_key('fookey')
    k.copy = mock.MagicMock()
    k.bucket = mock.MagicMock()
    k.bucket.name = 'mybucket'
    self.assertEqual(k.storage_class, 'STANDARD')
    k.change_storage_class('REDUCED_REDUNDANCY')
    k.copy.assert_called_with('mybucket', 'fookey', reduced_redundancy=True, preserve_acl=True, validate_dst_bucket=True)