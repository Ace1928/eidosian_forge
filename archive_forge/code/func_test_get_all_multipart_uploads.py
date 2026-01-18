import os
import unittest
import time
from boto.compat import StringIO
import mock
import boto
from boto.s3.connection import S3Connection
def test_get_all_multipart_uploads(self):
    key1 = 'a'
    key2 = 'b/c'
    mpu1 = self.bucket.initiate_multipart_upload(key1)
    mpu2 = self.bucket.initiate_multipart_upload(key2)
    rs = self.bucket.get_all_multipart_uploads(prefix='b/', delimiter='/')
    for lmpu in rs:
        self.assertEqual(lmpu.key_name, mpu2.key_name)
        self.assertEqual(lmpu.id, mpu2.id)