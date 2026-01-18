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
def test_get_item(self):
    expected = {'Item': {'username': {'S': 'johndoe'}, 'first_name': {'S': 'John'}, 'last_name': {'S': 'Doe'}, 'date_joined': {'N': '1366056668'}, 'friend_count': {'N': '3'}, 'friends': {'SS': ['alice', 'bob', 'jane']}}}
    with mock.patch.object(self.users.connection, 'get_item', return_value=expected) as mock_get_item:
        item = self.users.get_item(username='johndoe')
        self.assertEqual(item['username'], 'johndoe')
        self.assertEqual(item['first_name'], 'John')
    mock_get_item.assert_called_once_with('users', {'username': {'S': 'johndoe'}}, consistent_read=False, attributes_to_get=None)
    with mock.patch.object(self.users.connection, 'get_item', return_value=expected) as mock_get_item:
        item = self.users.get_item(username='johndoe', attributes=['username', 'first_name'])
    mock_get_item.assert_called_once_with('users', {'username': {'S': 'johndoe'}}, consistent_read=False, attributes_to_get=['username', 'first_name'])