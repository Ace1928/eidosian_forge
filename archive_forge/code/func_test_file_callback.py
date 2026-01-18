from tests.unit import unittest
import time
import random
import boto.s3
from boto.compat import six, StringIO, urllib
from boto.s3.connection import S3Connection
from boto.s3.key import Key
from boto.exception import S3ResponseError
def test_file_callback(self):

    def callback(wrote, total):
        self.my_cb_cnt += 1
        self.assertNotEqual(wrote, self.my_cb_last, 'called twice with same value')
        self.my_cb_last = wrote
    self.my_cb_cnt = 0
    self.my_cb_last = None
    k = self.bucket.new_key('k')
    k.BufferSize = 2
    sfp = StringIO('')
    k.set_contents_from_file(sfp, cb=callback, num_cb=10)
    self.assertEqual(self.my_cb_cnt, 1)
    self.assertEqual(self.my_cb_last, 0)
    sfp.close()
    self.my_cb_cnt = 0
    self.my_cb_last = None
    s = k.get_contents_as_string(cb=callback)
    self.assertEqual(self.my_cb_cnt, 1)
    self.assertEqual(self.my_cb_last, 0)
    content = '01234567890123456789'
    sfp = StringIO(content)
    self.my_cb_cnt = 0
    self.my_cb_last = None
    k = self.bucket.new_key('k')
    k.set_contents_from_file(sfp, cb=callback, num_cb=10)
    self.assertEqual(self.my_cb_cnt, 2)
    self.assertEqual(self.my_cb_last, 20)
    self.my_cb_cnt = 0
    self.my_cb_last = None
    s = k.get_contents_as_string(cb=callback).decode('utf-8')
    self.assertEqual(self.my_cb_cnt, 2)
    self.assertEqual(self.my_cb_last, 20)
    self.assertEqual(s, content)
    sfp.seek(0)
    self.my_cb_cnt = 0
    self.my_cb_last = None
    k = self.bucket.new_key('k')
    k.BufferSize = 2
    k.set_contents_from_file(sfp, cb=callback, num_cb=-1)
    self.assertEqual(self.my_cb_cnt, 11)
    self.assertEqual(self.my_cb_last, 20)
    self.my_cb_cnt = 0
    self.my_cb_last = None
    s = k.get_contents_as_string(cb=callback, num_cb=-1).decode('utf-8')
    self.assertEqual(self.my_cb_cnt, 11)
    self.assertEqual(self.my_cb_last, 20)
    self.assertEqual(s, content)
    sfp.seek(0)
    self.my_cb_cnt = 0
    self.my_cb_last = None
    k = self.bucket.new_key('k')
    k.BufferSize = 2
    k.set_contents_from_file(sfp, cb=callback, num_cb=1)
    self.assertTrue(self.my_cb_cnt <= 2)
    self.assertEqual(self.my_cb_last, 20)
    self.my_cb_cnt = 0
    self.my_cb_last = None
    s = k.get_contents_as_string(cb=callback, num_cb=1).decode('utf-8')
    self.assertTrue(self.my_cb_cnt <= 2)
    self.assertEqual(self.my_cb_last, 20)
    self.assertEqual(s, content)
    sfp.seek(0)
    self.my_cb_cnt = 0
    self.my_cb_last = None
    k = self.bucket.new_key('k')
    k.BufferSize = 2
    k.set_contents_from_file(sfp, cb=callback, num_cb=2)
    self.assertTrue(self.my_cb_cnt <= 2)
    self.assertEqual(self.my_cb_last, 20)
    self.my_cb_cnt = 0
    self.my_cb_last = None
    s = k.get_contents_as_string(cb=callback, num_cb=2).decode('utf-8')
    self.assertTrue(self.my_cb_cnt <= 2)
    self.assertEqual(self.my_cb_last, 20)
    self.assertEqual(s, content)
    sfp.seek(0)
    self.my_cb_cnt = 0
    self.my_cb_last = None
    k = self.bucket.new_key('k')
    k.BufferSize = 2
    k.set_contents_from_file(sfp, cb=callback, num_cb=3)
    self.assertTrue(self.my_cb_cnt <= 3)
    self.assertEqual(self.my_cb_last, 20)
    self.my_cb_cnt = 0
    self.my_cb_last = None
    s = k.get_contents_as_string(cb=callback, num_cb=3).decode('utf-8')
    self.assertTrue(self.my_cb_cnt <= 3)
    self.assertEqual(self.my_cb_last, 20)
    self.assertEqual(s, content)
    sfp.seek(0)
    self.my_cb_cnt = 0
    self.my_cb_last = None
    k = self.bucket.new_key('k')
    k.BufferSize = 2
    k.set_contents_from_file(sfp, cb=callback, num_cb=4)
    self.assertTrue(self.my_cb_cnt <= 4)
    self.assertEqual(self.my_cb_last, 20)
    self.my_cb_cnt = 0
    self.my_cb_last = None
    s = k.get_contents_as_string(cb=callback, num_cb=4).decode('utf-8')
    self.assertTrue(self.my_cb_cnt <= 4)
    self.assertEqual(self.my_cb_last, 20)
    self.assertEqual(s, content)
    sfp.seek(0)
    self.my_cb_cnt = 0
    self.my_cb_last = None
    k = self.bucket.new_key('k')
    k.BufferSize = 2
    k.set_contents_from_file(sfp, cb=callback, num_cb=6)
    self.assertTrue(self.my_cb_cnt <= 6)
    self.assertEqual(self.my_cb_last, 20)
    self.my_cb_cnt = 0
    self.my_cb_last = None
    s = k.get_contents_as_string(cb=callback, num_cb=6).decode('utf-8')
    self.assertTrue(self.my_cb_cnt <= 6)
    self.assertEqual(self.my_cb_last, 20)
    self.assertEqual(s, content)
    sfp.seek(0)
    self.my_cb_cnt = 0
    self.my_cb_last = None
    k = self.bucket.new_key('k')
    k.BufferSize = 2
    k.set_contents_from_file(sfp, cb=callback, num_cb=10)
    self.assertTrue(self.my_cb_cnt <= 10)
    self.assertEqual(self.my_cb_last, 20)
    self.my_cb_cnt = 0
    self.my_cb_last = None
    s = k.get_contents_as_string(cb=callback, num_cb=10).decode('utf-8')
    self.assertTrue(self.my_cb_cnt <= 10)
    self.assertEqual(self.my_cb_last, 20)
    self.assertEqual(s, content)
    sfp.seek(0)
    self.my_cb_cnt = 0
    self.my_cb_last = None
    k = self.bucket.new_key('k')
    k.BufferSize = 2
    k.set_contents_from_file(sfp, cb=callback, num_cb=1000)
    self.assertTrue(self.my_cb_cnt <= 1000)
    self.assertEqual(self.my_cb_last, 20)
    self.my_cb_cnt = 0
    self.my_cb_last = None
    s = k.get_contents_as_string(cb=callback, num_cb=1000).decode('utf-8')
    self.assertTrue(self.my_cb_cnt <= 1000)
    self.assertEqual(self.my_cb_last, 20)
    self.assertEqual(s, content)