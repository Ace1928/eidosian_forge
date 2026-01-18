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
def test_create_full(self):
    conn = FakeDynamoDBConnection()
    with mock.patch.object(conn, 'create_table', return_value={}) as mock_create_table:
        retval = Table.create('users', schema=[HashKey('username'), RangeKey('date_joined', data_type=NUMBER)], throughput={'read': 20, 'write': 10}, indexes=[KeysOnlyIndex('FriendCountIndex', parts=[RangeKey('friend_count')])], global_indexes=[GlobalKeysOnlyIndex('FullFriendCountIndex', parts=[RangeKey('friend_count')], throughput={'read': 10, 'write': 8})], connection=conn)
        self.assertTrue(retval)
    self.assertTrue(mock_create_table.called)
    mock_create_table.assert_called_once_with(attribute_definitions=[{'AttributeName': 'username', 'AttributeType': 'S'}, {'AttributeName': 'date_joined', 'AttributeType': 'N'}, {'AttributeName': 'friend_count', 'AttributeType': 'S'}], key_schema=[{'KeyType': 'HASH', 'AttributeName': 'username'}, {'KeyType': 'RANGE', 'AttributeName': 'date_joined'}], table_name='users', provisioned_throughput={'WriteCapacityUnits': 10, 'ReadCapacityUnits': 20}, global_secondary_indexes=[{'KeySchema': [{'KeyType': 'RANGE', 'AttributeName': 'friend_count'}], 'IndexName': 'FullFriendCountIndex', 'Projection': {'ProjectionType': 'KEYS_ONLY'}, 'ProvisionedThroughput': {'WriteCapacityUnits': 8, 'ReadCapacityUnits': 10}}], local_secondary_indexes=[{'KeySchema': [{'KeyType': 'RANGE', 'AttributeName': 'friend_count'}], 'IndexName': 'FriendCountIndex', 'Projection': {'ProjectionType': 'KEYS_ONLY'}}])