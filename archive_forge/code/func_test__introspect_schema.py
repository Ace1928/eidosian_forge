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
def test__introspect_schema(self):
    raw_schema_1 = [{'AttributeName': 'username', 'KeyType': 'HASH'}, {'AttributeName': 'date_joined', 'KeyType': 'RANGE'}]
    raw_attributes_1 = [{'AttributeName': 'username', 'AttributeType': 'S'}, {'AttributeName': 'date_joined', 'AttributeType': 'S'}]
    schema_1 = self.users._introspect_schema(raw_schema_1, raw_attributes_1)
    self.assertEqual(len(schema_1), 2)
    self.assertTrue(isinstance(schema_1[0], HashKey))
    self.assertEqual(schema_1[0].name, 'username')
    self.assertTrue(isinstance(schema_1[1], RangeKey))
    self.assertEqual(schema_1[1].name, 'date_joined')
    raw_schema_2 = [{'AttributeName': 'username', 'KeyType': 'BTREE'}]
    raw_attributes_2 = [{'AttributeName': 'username', 'AttributeType': 'S'}]
    self.assertRaises(exceptions.UnknownSchemaFieldError, self.users._introspect_schema, raw_schema_2, raw_attributes_2)
    raw_schema_3 = [{'AttributeName': 'user_id', 'KeyType': 'HASH'}, {'AttributeName': 'junk', 'KeyType': 'RANGE'}]
    raw_attributes_3 = [{'AttributeName': 'user_id', 'AttributeType': 'N'}, {'AttributeName': 'junk', 'AttributeType': 'B'}]
    schema_3 = self.users._introspect_schema(raw_schema_3, raw_attributes_3)
    self.assertEqual(len(schema_3), 2)
    self.assertTrue(isinstance(schema_3[0], HashKey))
    self.assertEqual(schema_3[0].name, 'user_id')
    self.assertEqual(schema_3[0].data_type, NUMBER)
    self.assertTrue(isinstance(schema_3[1], RangeKey))
    self.assertEqual(schema_3[1].name, 'junk')
    self.assertEqual(schema_3[1].data_type, BINARY)