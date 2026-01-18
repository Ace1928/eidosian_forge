import time
import boto
from tests.compat import unittest
from boto.kinesis.exceptions import ResourceNotFoundException
def test_describe_non_existent_stream(self):
    with self.assertRaises(ResourceNotFoundException) as cm:
        self.kinesis.describe_stream('this-stream-shouldnt-exist')
    self.assertEqual(cm.exception.error_code, None)
    self.assertTrue('not found' in cm.exception.message)