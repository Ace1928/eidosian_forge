import boto
from boto.kms.exceptions import NotFoundException
from tests.compat import unittest
def test_handle_not_found_exception(self):
    with self.assertRaises(NotFoundException):
        self.kms.describe_key(key_id='nonexistant_key')