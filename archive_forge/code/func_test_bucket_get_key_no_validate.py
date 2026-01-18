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
@patch.object(Bucket, 'get_all_keys')
@patch.object(Bucket, '_get_key_internal')
def test_bucket_get_key_no_validate(self, mock_gki, mock_gak):
    self.set_http_response(status_code=200)
    bucket = self.service_connection.get_bucket('mybucket')
    key = bucket.get_key('mykey', validate=False)
    self.assertEqual(len(mock_gki.mock_calls), 0)
    self.assertTrue(isinstance(key, Key))
    self.assertEqual(key.name, 'mykey')
    with self.assertRaises(BotoClientError):
        bucket.get_key('mykey', version_id='something', validate=False)