import boto
from boto.kms.exceptions import NotFoundException
from tests.compat import unittest
def test_list_keys(self):
    response = self.kms.list_keys()
    self.assertIn('Keys', response)