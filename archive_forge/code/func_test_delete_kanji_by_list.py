import unittest
import time
from boto.s3.key import Key
from boto.s3.deletemarker import DeleteMarker
from boto.s3.prefix import Prefix
from boto.s3.connection import S3Connection
from boto.exception import S3ResponseError
def test_delete_kanji_by_list(self):
    for key_name in [u'漢字', u'日本語', u'テスト']:
        key = self.bucket.new_key(key_name)
        key.set_contents_from_string('this is a test')
    result = self.bucket.delete_keys(self.bucket.list())
    self.assertEqual(len(result.deleted), 3)
    self.assertEqual(len(result.errors), 0)