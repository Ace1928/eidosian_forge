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
@patch.object(Bucket, '_get_all')
def test_bucket_encoding(self, mock_get_all):
    self.set_http_response(status_code=200)
    bucket = self.service_connection.get_bucket('mybucket')
    mock_get_all.reset_mock()
    bucket.get_all_keys()
    mock_get_all.assert_called_with([('Contents', Key), ('CommonPrefixes', Prefix)], '', None)
    mock_get_all.reset_mock()
    bucket.get_all_keys(encoding_type='url')
    mock_get_all.assert_called_with([('Contents', Key), ('CommonPrefixes', Prefix)], '', None, encoding_type='url')
    mock_get_all.reset_mock()
    bucket.get_all_versions(encoding_type='url')
    mock_get_all.assert_called_with([('Version', Key), ('CommonPrefixes', Prefix), ('DeleteMarker', DeleteMarker)], 'versions', None, encoding_type='url')
    mock_get_all.reset_mock()
    bucket.get_all_multipart_uploads(encoding_type='url')
    mock_get_all.assert_called_with([('Upload', MultiPartUpload), ('CommonPrefixes', Prefix)], 'uploads', None, encoding_type='url')