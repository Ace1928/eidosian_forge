from tests.compat import mock, unittest
from boto.dynamodb2 import exceptions
from boto.dynamodb2.fields import (HashKey, RangeKey,
from boto.dynamodb2.items import Item
from boto.dynamodb2.layer1 import DynamoDBConnection
from boto.dynamodb2.results import ResultSet, BatchGetResultSet
from boto.dynamodb2.table import Table
from boto.dynamodb2.types import (STRING, NUMBER, BINARY,
from boto.exception import JSONResponseError
from boto.compat import six, long_type
def test_build_expects(self):
    self.assertEqual(self.johndoe.build_expects(), {'first_name': {'Exists': False}, 'username': {'Exists': False}, 'date_joined': {'Exists': False}})
    self.johndoe.mark_clean()
    self.assertEqual(self.johndoe.build_expects(), {'first_name': {'Exists': True, 'Value': {'S': 'John'}}, 'username': {'Exists': True, 'Value': {'S': 'johndoe'}}, 'date_joined': {'Exists': True, 'Value': {'N': '12345'}}})
    self.johndoe['first_name'] = 'Johann'
    self.johndoe['last_name'] = 'Doe'
    del self.johndoe['date_joined']
    self.assertEqual(self.johndoe.build_expects(), {'first_name': {'Exists': True, 'Value': {'S': 'John'}}, 'last_name': {'Exists': False}, 'username': {'Exists': True, 'Value': {'S': 'johndoe'}}, 'date_joined': {'Exists': True, 'Value': {'N': '12345'}}})
    self.assertEqual(self.johndoe.build_expects(fields=['first_name', 'last_name', 'date_joined']), {'first_name': {'Exists': True, 'Value': {'S': 'John'}}, 'last_name': {'Exists': False}, 'date_joined': {'Exists': True, 'Value': {'N': '12345'}}})