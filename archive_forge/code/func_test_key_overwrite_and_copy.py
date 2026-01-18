from tests.unit import unittest
import time
import random
import boto.s3
from boto.compat import six, StringIO, urllib
from boto.s3.connection import S3Connection
from boto.s3.key import Key
from boto.exception import S3ResponseError
def test_key_overwrite_and_copy(self):
    first_content = b'abcdefghijklm'
    second_content = b'nopqrstuvwxyz'
    k = Key(self.bucket, 'testkey')
    k.set_contents_from_string(first_content)
    while self.bucket.get_key('testkey') is None:
        time.sleep(5)
    first_key = self.bucket.get_key('testkey')
    first_version_id = first_key.version_id
    k = Key(self.bucket, 'testkey')
    k.set_contents_from_string(second_content)
    while True:
        second_key = self.bucket.get_key('testkey')
        if second_key is None or second_key.version_id == first_version_id:
            time.sleep(5)
        else:
            break
    source_key = self.bucket.get_key('testkey', version_id=first_version_id)
    source_key.copy(self.bucket, 'copiedkey')
    while self.bucket.get_key('copiedkey') is None:
        time.sleep(5)
    copied_key = self.bucket.get_key('copiedkey')
    copied_key_contents = copied_key.get_contents_as_string()
    self.assertEqual(first_content, copied_key_contents)