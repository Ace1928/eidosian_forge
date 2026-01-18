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
def test_get_key_fields_no_schema_populated(self):
    expected = {'Table': {'AttributeDefinitions': [{'AttributeName': 'username', 'AttributeType': 'S'}, {'AttributeName': 'date_joined', 'AttributeType': 'N'}], 'ItemCount': 5, 'KeySchema': [{'AttributeName': 'username', 'KeyType': 'HASH'}, {'AttributeName': 'date_joined', 'KeyType': 'RANGE'}], 'LocalSecondaryIndexes': [{'IndexName': 'UsernameIndex', 'KeySchema': [{'AttributeName': 'username', 'KeyType': 'HASH'}], 'Projection': {'ProjectionType': 'KEYS_ONLY'}}], 'ProvisionedThroughput': {'ReadCapacityUnits': 20, 'WriteCapacityUnits': 6}, 'TableName': 'Thread', 'TableStatus': 'ACTIVE'}}
    with mock.patch.object(self.users.connection, 'describe_table', return_value=expected) as mock_describe:
        self.assertEqual(self.users.schema, None)
        key_fields = self.users.get_key_fields()
        self.assertEqual(key_fields, ['username', 'date_joined'])
        self.assertEqual(len(self.users.schema), 2)
    mock_describe.assert_called_once_with('users')