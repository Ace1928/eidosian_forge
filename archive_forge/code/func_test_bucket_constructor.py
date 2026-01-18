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
def test_bucket_constructor(self):
    self.set_http_response(status_code=200)
    bucket = Bucket(self.service_connection, 'mybucket_constructor')
    self.assertEqual(bucket.name, 'mybucket_constructor')