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
def test__introspect_indexes(self):
    raw_indexes_1 = [{'IndexName': 'MostRecentlyJoinedIndex', 'KeySchema': [{'AttributeName': 'username', 'KeyType': 'HASH'}, {'AttributeName': 'date_joined', 'KeyType': 'RANGE'}], 'Projection': {'ProjectionType': 'KEYS_ONLY'}}, {'IndexName': 'EverybodyIndex', 'KeySchema': [{'AttributeName': 'username', 'KeyType': 'HASH'}], 'Projection': {'ProjectionType': 'ALL'}}, {'IndexName': 'GenderIndex', 'KeySchema': [{'AttributeName': 'username', 'KeyType': 'HASH'}, {'AttributeName': 'date_joined', 'KeyType': 'RANGE'}], 'Projection': {'ProjectionType': 'INCLUDE', 'NonKeyAttributes': ['gender']}}]
    indexes_1 = self.users._introspect_indexes(raw_indexes_1)
    self.assertEqual(len(indexes_1), 3)
    self.assertTrue(isinstance(indexes_1[0], KeysOnlyIndex))
    self.assertEqual(indexes_1[0].name, 'MostRecentlyJoinedIndex')
    self.assertEqual(len(indexes_1[0].parts), 2)
    self.assertTrue(isinstance(indexes_1[1], AllIndex))
    self.assertEqual(indexes_1[1].name, 'EverybodyIndex')
    self.assertEqual(len(indexes_1[1].parts), 1)
    self.assertTrue(isinstance(indexes_1[2], IncludeIndex))
    self.assertEqual(indexes_1[2].name, 'GenderIndex')
    self.assertEqual(len(indexes_1[2].parts), 2)
    self.assertEqual(indexes_1[2].includes_fields, ['gender'])
    raw_indexes_2 = [{'IndexName': 'MostRecentlyJoinedIndex', 'KeySchema': [{'AttributeName': 'username', 'KeyType': 'HASH'}, {'AttributeName': 'date_joined', 'KeyType': 'RANGE'}], 'Projection': {'ProjectionType': 'SOMETHING_CRAZY'}}]
    self.assertRaises(exceptions.UnknownIndexFieldError, self.users._introspect_indexes, raw_indexes_2)