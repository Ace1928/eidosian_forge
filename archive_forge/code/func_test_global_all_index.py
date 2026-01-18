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
def test_global_all_index(self):
    all_index = GlobalAllIndex('AllKeys', parts=[HashKey('username'), RangeKey('date_joined')], throughput={'read': 6, 'write': 2})
    self.assertEqual(all_index.name, 'AllKeys')
    self.assertEqual([part.attr_type for part in all_index.parts], ['HASH', 'RANGE'])
    self.assertEqual(all_index.projection_type, 'ALL')
    self.assertEqual(all_index.definition(), [{'AttributeName': 'username', 'AttributeType': 'S'}, {'AttributeName': 'date_joined', 'AttributeType': 'S'}])
    self.assertEqual(all_index.schema(), {'IndexName': 'AllKeys', 'KeySchema': [{'AttributeName': 'username', 'KeyType': 'HASH'}, {'AttributeName': 'date_joined', 'KeyType': 'RANGE'}], 'Projection': {'ProjectionType': 'ALL'}, 'ProvisionedThroughput': {'ReadCapacityUnits': 6, 'WriteCapacityUnits': 2}})