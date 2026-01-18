import unittest
import time
from boto.s3.key import Key
from boto.s3.deletemarker import DeleteMarker
from boto.s3.prefix import Prefix
from boto.s3.connection import S3Connection
from boto.exception import S3ResponseError
def test_delete_with_prefixes(self):
    for key_name in ['a', 'a/b', 'b']:
        key = self.bucket.new_key(key_name)
        key.set_contents_from_string('this is a test')
    result = self.bucket.delete_keys(self.bucket.list(delimiter='/'))
    self.assertEqual(len(result.deleted), 2)
    self.assertEqual(len(result.errors), 1)
    self.assertEqual(result.errors[0].key, 'a/')
    result = self.bucket.delete_keys(self.bucket.list())
    self.assertEqual(len(result.deleted), 1)
    self.assertEqual(len(result.errors), 0)
    self.assertEqual(result.deleted[0].key, 'a/b')