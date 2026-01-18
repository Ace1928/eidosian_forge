from decimal import Decimal
from tests.compat import unittest
from boto.compat import six, json
from boto.dynamodb import types
from boto.dynamodb.exceptions import DynamoDBNumberError
def test_lossy_float_conversions(self):
    dynamizer = types.LossyFloatDynamizer()
    self.assertEqual(dynamizer.encode(1.1), {'N': '1.1'})
    self.assertEqual(dynamizer.decode({'N': '1.1'}), 1.1)
    self.assertEqual(dynamizer.encode(set([1.1])), {'NS': ['1.1']})
    self.assertEqual(dynamizer.decode({'NS': ['1.1', '2.2', '3.3']}), set([1.1, 2.2, 3.3]))