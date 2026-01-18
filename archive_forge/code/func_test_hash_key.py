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
def test_hash_key(self):
    hash_key = HashKey('hello')
    self.assertEqual(hash_key.name, 'hello')
    self.assertEqual(hash_key.data_type, STRING)
    self.assertEqual(hash_key.attr_type, 'HASH')
    self.assertEqual(hash_key.definition(), {'AttributeName': 'hello', 'AttributeType': 'S'})
    self.assertEqual(hash_key.schema(), {'AttributeName': 'hello', 'KeyType': 'HASH'})