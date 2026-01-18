from decimal import Decimal
from tests.compat import unittest
from boto.compat import six, json
from boto.dynamodb import types
from boto.dynamodb.exceptions import DynamoDBNumberError
def test_float_conversion_errors(self):
    dynamizer = types.Dynamizer()
    self.assertEqual(dynamizer.encode(1.25), {'N': '1.25'})
    with self.assertRaises(DynamoDBNumberError):
        dynamizer.encode(1.1)