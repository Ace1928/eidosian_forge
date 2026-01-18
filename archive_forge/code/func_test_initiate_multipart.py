import os
import unittest
import time
from boto.compat import StringIO
import mock
import boto
from boto.s3.connection import S3Connection
def test_initiate_multipart(self):
    key_name = 'multipart'
    multipart_upload = self.bucket.initiate_multipart_upload(key_name)
    multipart_uploads = self.bucket.get_all_multipart_uploads()
    for upload in multipart_uploads:
        self.assertEqual(upload.key_name, multipart_upload.key_name)
        self.assertEqual(upload.id, multipart_upload.id)
    multipart_upload.cancel_upload()