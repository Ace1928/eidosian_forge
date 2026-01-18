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
def test_bucket_acl_policy_namespace(self):
    self.set_http_response(status_code=200)
    bucket = self.service_connection.get_bucket('mybucket')
    self.set_http_response(status_code=200, body=self.acl_policy())
    policy = bucket.get_acl()
    xml_policy = policy.to_xml()
    document = xml.dom.minidom.parseString(xml_policy)
    namespace = document.documentElement.namespaceURI
    self.assertEqual(namespace, 'http://s3.amazonaws.com/doc/2006-03-01/')