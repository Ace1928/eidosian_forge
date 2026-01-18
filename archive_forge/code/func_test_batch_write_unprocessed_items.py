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
def test_batch_write_unprocessed_items(self):
    unprocessed = {'UnprocessedItems': {'users': [{'PutRequest': {'username': {'S': 'jane'}, 'date_joined': {'N': 12342547}}}]}}
    with mock.patch.object(self.users.connection, 'batch_write_item', return_value=unprocessed) as mock_batch:
        with self.users.batch_write() as batch:
            self.assertEqual(len(batch._unprocessed), 0)
            batch.resend_unprocessed = lambda: True
            batch.put_item(data={'username': 'jane', 'date_joined': 12342547})
            batch.delete_item(username='johndoe')
            batch.put_item(data={'username': 'alice', 'date_joined': 12342888})
        self.assertEqual(len(batch._unprocessed), 1)
    with mock.patch.object(self.users.connection, 'batch_write_item', return_value={}) as mock_batch:
        with self.users.batch_write() as batch:
            self.assertEqual(len(batch._unprocessed), 0)
            batch._unprocessed = [{'PutRequest': {'username': {'S': 'jane'}, 'date_joined': {'N': 12342547}}}]
            batch.put_item(data={'username': 'jane', 'date_joined': 12342547})
            batch.delete_item(username='johndoe')
            batch.put_item(data={'username': 'alice', 'date_joined': 12342888})
            batch.flush()
            self.assertEqual(len(batch._unprocessed), 1)
        self.assertEqual(len(batch._unprocessed), 0)