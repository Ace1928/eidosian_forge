import os
import unittest
import time
from boto.compat import StringIO
import mock
import boto
from boto.s3.connection import S3Connection
def test_list_multipart_uploads(self):
    key_name = u'テスト'
    mpus = []
    mpus.append(self.bucket.initiate_multipart_upload(key_name))
    mpus.append(self.bucket.initiate_multipart_upload(key_name))
    rs = self.bucket.list_multipart_uploads()
    for lmpu in rs:
        ompu = mpus.pop(0)
        self.assertEqual(lmpu.key_name, ompu.key_name)
        self.assertEqual(lmpu.id, ompu.id)
    self.assertEqual(0, len(mpus))