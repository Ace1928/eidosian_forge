import boto
from boto.awslambda.exceptions import ResourceNotFoundException
from tests.compat import unittest
def test_list_functions(self):
    response = self.awslambda.list_functions()
    self.assertIn('Functions', response)