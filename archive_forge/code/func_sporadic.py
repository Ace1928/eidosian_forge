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
def sporadic():
    count = {'value': -1}

    def _wrapper(limit=10, exclusive_start_key=None):
        count['value'] = count['value'] + 1
        if count['value'] == 0:
            return {'results': ['Result #0', 'Result #1', 'Result #2', 'Result #3'], 'last_key': 'page-1'}
        elif count['value'] == 1:
            return {'results': [], 'last_key': 'page-2'}
        elif count['value'] == 2:
            return {'results': ['Result #4', 'Result #5', 'Result #6']}
    return _wrapper