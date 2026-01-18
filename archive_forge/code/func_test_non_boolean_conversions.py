from decimal import Decimal
from tests.compat import unittest
from boto.compat import six, json
from boto.dynamodb import types
from boto.dynamodb.exceptions import DynamoDBNumberError
def test_non_boolean_conversions(self):
    dynamizer = types.NonBooleanDynamizer()
    self.assertEqual(dynamizer.encode(True), {'N': '1'})