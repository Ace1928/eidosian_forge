from mock import patch, Mock
import unittest
import time
from boto.exception import S3ResponseError
from boto.s3.connection import S3Connection
from boto.s3.bucketlogging import BucketLogging
from boto.s3.lifecycle import Lifecycle
from boto.s3.lifecycle import Transition
from boto.s3.lifecycle import Expiration
from boto.s3.lifecycle import Rule
from boto.s3.acl import Grant
from boto.s3.tagging import Tags, TagSet
from boto.s3.website import RedirectLocation
from boto.compat import unquote_str
def test_next_marker(self):
    expected = ['a/', 'b', 'c']
    for key_name in expected:
        key = self.bucket.new_key(key_name)
        key.set_contents_from_string(key_name)
    rs = self.bucket.get_all_keys(max_keys=2)
    for element in rs:
        pass
    self.assertEqual(element.name, 'b')
    self.assertEqual(rs.next_marker, None)
    rs = self.bucket.get_all_keys(max_keys=2, delimiter='/')
    for element in rs:
        pass
    self.assertEqual(element.name, 'a/')
    self.assertEqual(rs.next_marker, 'b')
    rs = self.bucket.list()
    for element in rs:
        self.assertEqual(element.name, expected.pop(0))
    self.assertEqual(expected, [])