import unittest
import time
from boto.s3.connection import S3Connection
from boto.exception import S3ResponseError
def test_1_versions(self):
    print('--- running S3Encryption tests ---')
    c = S3Connection()
    bucket_name = 'encryption-%d' % int(time.time())
    bucket = c.create_bucket(bucket_name)
    bucket = c.get_bucket(bucket_name)
    k = bucket.new_key('foobar')
    s1 = 'This is unencrypted data'
    s2 = 'This is encrypted data'
    k.set_contents_from_string(s1)
    time.sleep(5)
    o = k.get_contents_as_string().decode('utf-8')
    assert o == s1
    k.set_contents_from_string(s2, encrypt_key=True)
    time.sleep(5)
    o = k.get_contents_as_string().decode('utf-8')
    assert o == s2
    bucket.set_policy(json_policy % bucket.name)
    time.sleep(5)
    write_failed = False
    try:
        k.set_contents_from_string(s1)
    except S3ResponseError:
        write_failed = True
    assert write_failed
    write_failed = False
    try:
        k.set_contents_from_string(s1, encrypt_key=True)
    except S3ResponseError:
        write_failed = True
    assert not write_failed
    k.delete()
    time.sleep(5)
    bucket.delete()
    print('--- tests completed ---')