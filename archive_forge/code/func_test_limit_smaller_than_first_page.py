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
def test_limit_smaller_than_first_page(self):
    results = ResultSet()
    results.to_call(fake_results, 'john', greeting='Hello', limit=2)
    self.assertEqual(next(results), 'Hello john #0')
    self.assertEqual(next(results), 'Hello john #1')
    self.assertRaises(StopIteration, results.next)