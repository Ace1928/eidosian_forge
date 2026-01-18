import datetime
import pytz
from graphql import GraphQLError
from pytest import fixture
from ..datetime import Date, DateTime, Time
from ..objecttype import ObjectType
from ..schema import Schema
def test_bad_datetime_query():
    not_a_date = "Some string that's not a datetime"
    result = schema.execute('{ datetime(in: "%s") }' % not_a_date)
    assert result.errors and len(result.errors) == 1
    error = result.errors[0]
    assert isinstance(error, GraphQLError)
    assert error.message == 'DateTime cannot represent value: "Some string that\'s not a datetime"'
    assert result.data is None