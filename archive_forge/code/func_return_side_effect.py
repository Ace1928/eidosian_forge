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
def return_side_effect(*args, **kwargs):
    if kwargs.get('exclusive_start_key'):
        return {'Count': 10, 'LastEvaluatedKey': None}
    else:
        return {'Count': 20, 'LastEvaluatedKey': {'username': {'S': 'johndoe'}, 'date_joined': {'N': '4118642633'}}}