import os
import unittest
import time
from boto.compat import StringIO
import mock
import boto
from boto.s3.connection import S3Connection
def test_complete_japanese(self):
    key_name = u'テスト'
    mpu = self.bucket.initiate_multipart_upload(key_name)
    fp = StringIO('small file')
    mpu.upload_part_from_file(fp, part_num=1)
    fp.close()
    cmpu = mpu.complete_upload()
    self.assertEqual(cmpu.key_name, key_name)
    self.assertNotEqual(cmpu.etag, None)