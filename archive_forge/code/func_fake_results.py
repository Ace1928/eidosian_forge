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
def fake_results(name, greeting='hello', exclusive_start_key=None, limit=None):
    if exclusive_start_key is None:
        exclusive_start_key = -1
    if limit == 0:
        raise Exception("Web Service Returns '400 Bad Request'")
    end_cap = 13
    results = []
    start_key = exclusive_start_key + 1
    for i in range(start_key, start_key + 5):
        if i < end_cap:
            results.append('%s %s #%s' % (greeting, name, i))
    if limit < len(results):
        results = results[:limit]
    retval = {'results': results}
    if exclusive_start_key + 5 < end_cap:
        retval['last_key'] = exclusive_start_key + 5
    return retval