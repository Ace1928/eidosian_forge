import unittest
import time
import os
import socket
from boto.s3.connection import S3Connection
from boto.s3.bucket import Bucket
from boto.exception import S3PermissionsError, S3ResponseError
from boto.compat import http_client, six, urlopen, urlsplit
def test_error_code_populated(self):
    c = S3Connection()
    try:
        c.create_bucket('bad$bucket$name')
    except S3ResponseError as e:
        self.assertEqual(e.error_code, 'InvalidBucketName')
    except socket.gaierror:
        pass
    else:
        self.fail('S3ResponseError not raised.')