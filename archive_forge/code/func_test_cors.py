import unittest
import time
from boto.s3.connection import S3Connection
from boto.exception import S3ResponseError
from boto.s3.cors import CORSConfiguration
def test_cors(self):
    self.cfg = CORSConfiguration()
    self.cfg.add_rule(['PUT', 'POST', 'DELETE'], 'http://www.example.com', allowed_header='*', max_age_seconds=3000, expose_header='x-amz-server-side-encryption', id='foobar_rule')
    assert self.bucket.set_cors(self.cfg)
    time.sleep(5)
    cfg = self.bucket.get_cors()
    for i, rule in enumerate(cfg):
        self.assertEqual(rule.id, self.cfg[i].id)
        self.assertEqual(rule.max_age_seconds, self.cfg[i].max_age_seconds)
        methods = zip(rule.allowed_method, self.cfg[i].allowed_method)
        for v1, v2 in methods:
            self.assertEqual(v1, v2)
        origins = zip(rule.allowed_origin, self.cfg[i].allowed_origin)
        for v1, v2 in origins:
            self.assertEqual(v1, v2)
        headers = zip(rule.allowed_header, self.cfg[i].allowed_header)
        for v1, v2 in headers:
            self.assertEqual(v1, v2)
        headers = zip(rule.expose_header, self.cfg[i].expose_header)
        for v1, v2 in headers:
            self.assertEqual(v1, v2)
    self.bucket.delete_cors()
    time.sleep(5)
    try:
        self.bucket.get_cors()
        self.fail('CORS configuration should not be there')
    except S3ResponseError:
        pass