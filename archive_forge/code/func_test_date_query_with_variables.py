import datetime
import pytz
from graphql import GraphQLError
from pytest import fixture
from ..datetime import Date, DateTime, Time
from ..objecttype import ObjectType
from ..schema import Schema
def test_date_query_with_variables(sample_date):
    isoformat = sample_date.isoformat()
    result = schema.execute('\n        query GetDate($date: Date) {\n          literal: date(in: "%s")\n          value: date(in: $date)\n        }\n        ' % isoformat, variable_values={'date': isoformat})
    assert not result.errors
    assert result.data == {'literal': isoformat, 'value': isoformat}