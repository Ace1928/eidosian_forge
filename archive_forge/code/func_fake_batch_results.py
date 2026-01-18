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
def fake_batch_results(keys):
    results = []
    simulate_unprocessed = True
    if len(keys) and keys[0] == 'johndoe':
        simulate_unprocessed = False
    for key in keys:
        if simulate_unprocessed and key == 'johndoe':
            continue
        results.append('hello %s' % key)
    retval = {'results': results, 'last_key': None}
    if simulate_unprocessed:
        retval['unprocessed_keys'] = ['johndoe']
    return retval