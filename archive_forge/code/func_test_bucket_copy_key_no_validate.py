from mock import patch
import xml.dom.minidom
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.exception import BotoClientError
from boto.s3.connection import Location, S3Connection
from boto.s3.bucket import Bucket
from boto.s3.deletemarker import DeleteMarker
from boto.s3.key import Key
from boto.s3.multipart import MultiPartUpload
from boto.s3.prefix import Prefix
@patch.object(S3Connection, 'head_bucket')
def test_bucket_copy_key_no_validate(self, mock_head_bucket):
    self.set_http_response(status_code=200)
    bucket = self.service_connection.create_bucket('mybucket')
    self.assertFalse(mock_head_bucket.called)
    self.service_connection.get_bucket('mybucket', validate=True)
    self.assertTrue(mock_head_bucket.called)
    mock_head_bucket.reset_mock()
    self.assertFalse(mock_head_bucket.called)
    try:
        bucket.copy_key('newkey', 'srcbucket', 'srckey', preserve_acl=True)
    except:
        pass
    self.assertFalse(mock_head_bucket.called)