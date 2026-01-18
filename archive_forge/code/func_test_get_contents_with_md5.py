from tests.unit import unittest
import time
import random
import boto.s3
from boto.compat import six, StringIO, urllib
from boto.s3.connection import S3Connection
from boto.s3.key import Key
from boto.exception import S3ResponseError
def test_get_contents_with_md5(self):
    content = '01234567890123456789'
    sfp = StringIO(content)
    k = self.bucket.new_key('k')
    k.set_contents_from_file(sfp)
    kn = self.bucket.new_key('k')
    s = kn.get_contents_as_string().decode('utf-8')
    self.assertEqual(kn.md5, k.md5)
    self.assertEqual(s, content)