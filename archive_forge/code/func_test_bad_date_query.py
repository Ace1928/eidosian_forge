import datetime
import pytz
from graphql import GraphQLError
from pytest import fixture
from ..datetime import Date, DateTime, Time
from ..objecttype import ObjectType
from ..schema import Schema
def test_bad_date_query():
    not_a_date = "Some string that's not a date"
    result = schema.execute('{ date(in: "%s") }' % not_a_date)
    error = result.errors[0]
    assert isinstance(error, GraphQLError)
    assert error.message == 'Date cannot represent value: "Some string that\'s not a date"'
    assert result.data is None