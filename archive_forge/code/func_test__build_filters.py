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
def test__build_filters(self):
    filters = self.users._build_filters({'username__eq': 'johndoe', 'date_joined__gte': 1234567, 'age__in': [30, 31, 32, 33], 'last_name__between': ['danzig', 'only'], 'first_name__null': False, 'gender__null': True}, using=FILTER_OPERATORS)
    self.assertEqual(filters, {'username': {'AttributeValueList': [{'S': 'johndoe'}], 'ComparisonOperator': 'EQ'}, 'date_joined': {'AttributeValueList': [{'N': '1234567'}], 'ComparisonOperator': 'GE'}, 'age': {'AttributeValueList': [{'N': '30'}, {'N': '31'}, {'N': '32'}, {'N': '33'}], 'ComparisonOperator': 'IN'}, 'last_name': {'AttributeValueList': [{'S': 'danzig'}, {'S': 'only'}], 'ComparisonOperator': 'BETWEEN'}, 'first_name': {'ComparisonOperator': 'NOT_NULL'}, 'gender': {'ComparisonOperator': 'NULL'}})
    self.assertRaises(exceptions.UnknownFilterTypeError, self.users._build_filters, {'darling__die': True})
    q_filters = self.users._build_filters({'username__eq': 'johndoe', 'date_joined__gte': 1234567, 'last_name__between': ['danzig', 'only'], 'gender__beginswith': 'm'}, using=QUERY_OPERATORS)
    self.assertEqual(q_filters, {'username': {'AttributeValueList': [{'S': 'johndoe'}], 'ComparisonOperator': 'EQ'}, 'date_joined': {'AttributeValueList': [{'N': '1234567'}], 'ComparisonOperator': 'GE'}, 'last_name': {'AttributeValueList': [{'S': 'danzig'}, {'S': 'only'}], 'ComparisonOperator': 'BETWEEN'}, 'gender': {'AttributeValueList': [{'S': 'm'}], 'ComparisonOperator': 'BEGINS_WITH'}})
    self.assertRaises(exceptions.UnknownFilterTypeError, self.users._build_filters, {'darling__die': True}, using=QUERY_OPERATORS)
    self.assertRaises(exceptions.UnknownFilterTypeError, self.users._build_filters, {'first_name__null': True}, using=QUERY_OPERATORS)