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
def test_batch_get(self):
    items_1 = {'results': [Item(self.users, data={'username': 'johndoe', 'first_name': 'John', 'last_name': 'Doe'}), Item(self.users, data={'username': 'jane', 'first_name': 'Jane', 'last_name': 'Doe'})], 'last_key': None, 'unprocessed_keys': ['zoeydoe']}
    results = self.users.batch_get(keys=[{'username': 'johndoe'}, {'username': 'jane'}, {'username': 'zoeydoe'}])
    self.assertTrue(isinstance(results, BatchGetResultSet))
    self.assertEqual(len(results._results), 0)
    self.assertEqual(results.the_callable, self.users._batch_get)
    with mock.patch.object(results, 'the_callable', return_value=items_1) as mock_batch_get:
        res_1 = next(results)
        self.assertEqual(len(results._results), 2)
        self.assertEqual(res_1['username'], 'johndoe')
        res_2 = next(results)
        self.assertEqual(res_2['username'], 'jane')
    self.assertEqual(mock_batch_get.call_count, 1)
    self.assertEqual(results._keys_left, ['zoeydoe'])
    items_2 = {'results': [Item(self.users, data={'username': 'zoeydoe', 'first_name': 'Zoey', 'last_name': 'Doe'})]}
    with mock.patch.object(results, 'the_callable', return_value=items_2) as mock_batch_get_2:
        res_3 = next(results)
        self.assertEqual(len(results._results), 1)
        self.assertEqual(res_3['username'], 'zoeydoe')
        self.assertRaises(StopIteration, results.next)
    self.assertEqual(mock_batch_get_2.call_count, 1)
    self.assertEqual(results._keys_left, [])