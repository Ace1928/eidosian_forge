from tests.unit import unittest
import time
import random
import boto.s3
from boto.compat import six, StringIO, urllib
from boto.s3.connection import S3Connection
from boto.s3.key import Key
from boto.exception import S3ResponseError
def test_header_encoding(self):
    key = self.bucket.new_key('test_header_encoding')
    key.set_metadata('Cache-control', u'public, max-age=500')
    key.set_metadata('Test-Plus', u'A plus (+)')
    key.set_metadata('Content-disposition', u'filename=Schöne Zeit.txt')
    key.set_metadata('Content-Encoding', 'gzip')
    key.set_metadata('Content-Language', 'de')
    key.set_metadata('Content-Type', 'application/pdf')
    self.assertEqual(key.content_type, 'application/pdf')
    key.set_metadata('X-Robots-Tag', 'all')
    key.set_metadata('Expires', u'Thu, 01 Dec 1994 16:00:00 GMT')
    key.set_contents_from_string('foo')
    check = self.bucket.get_key('test_header_encoding')
    remote_metadata = check._get_remote_metadata()
    self.assertIn(check.cache_control, ('public,%20max-age=500', 'public, max-age=500'))
    self.assertIn(remote_metadata['cache-control'], ('public,%20max-age=500', 'public, max-age=500'))
    self.assertEqual(check.get_metadata('test-plus'), 'A plus (+)')
    self.assertEqual(check.content_disposition, 'filename=Sch%C3%B6ne Zeit.txt')
    self.assertEqual(remote_metadata['content-disposition'], 'filename=Sch%C3%B6ne Zeit.txt')
    self.assertEqual(check.content_encoding, 'gzip')
    self.assertEqual(remote_metadata['content-encoding'], 'gzip')
    self.assertEqual(check.content_language, 'de')
    self.assertEqual(remote_metadata['content-language'], 'de')
    self.assertEqual(check.content_type, 'application/pdf')
    self.assertEqual(remote_metadata['content-type'], 'application/pdf')
    self.assertEqual(check.x_robots_tag, 'all')
    self.assertEqual(remote_metadata['x-robots-tag'], 'all')
    self.assertEqual(check.expires, 'Thu, 01 Dec 1994 16:00:00 GMT')
    self.assertEqual(remote_metadata['expires'], 'Thu, 01 Dec 1994 16:00:00 GMT')
    expected = u'filename=Schöne Zeit.txt'
    if six.PY2:
        expected = expected.encode('utf-8')
    self.assertEqual(urllib.parse.unquote(check.content_disposition), expected)