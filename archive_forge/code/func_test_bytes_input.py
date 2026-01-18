from decimal import Decimal
from tests.compat import unittest
from boto.compat import six, json
from boto.dynamodb import types
from boto.dynamodb.exceptions import DynamoDBNumberError
@unittest.skipUnless(six.PY3, 'Python 3 only')
def test_bytes_input(self):
    data = types.Binary(1)
    self.assertEqual(data, b'\x00')
    self.assertEqual(data.value, b'\x00')