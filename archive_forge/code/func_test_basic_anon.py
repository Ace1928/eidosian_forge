import unittest
import time
import os
import socket
from boto.s3.connection import S3Connection
from boto.s3.bucket import Bucket
from boto.exception import S3PermissionsError, S3ResponseError
from boto.compat import http_client, six, urlopen, urlsplit
def test_basic_anon(self):
    auth_con = S3Connection()
    bucket_name = 'test-%d' % int(time.time())
    auth_bucket = auth_con.create_bucket(bucket_name)
    anon_con = S3Connection(anon=True)
    anon_bucket = Bucket(anon_con, bucket_name)
    try:
        next(iter(anon_bucket.list()))
        self.fail('anon bucket list should fail')
    except S3ResponseError:
        pass
    auth_bucket.set_acl('public-read')
    time.sleep(10)
    try:
        next(iter(anon_bucket.list()))
        self.fail('not expecting contents')
    except S3ResponseError as e:
        self.fail('We should have public-read access, but received an error: %s' % e)
    except StopIteration:
        pass
    auth_con.delete_bucket(auth_bucket)