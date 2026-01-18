import os
import unittest
import time
from boto.compat import StringIO
import mock
import boto
from boto.s3.connection import S3Connection
def test_list_japanese(self):
    key_name = u'テスト'
    mpu = self.bucket.initiate_multipart_upload(key_name)
    rs = self.bucket.list_multipart_uploads()
    lmpu = next(iter(rs))
    self.assertEqual(lmpu.id, mpu.id)
    self.assertEqual(lmpu.key_name, key_name)
    lmpu.cancel_upload()