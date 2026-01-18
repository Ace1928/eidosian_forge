import datetime
import pytz
from graphql import GraphQLError
from pytest import fixture
from ..datetime import Date, DateTime, Time
from ..objecttype import ObjectType
from ..schema import Schema
def test_date_query_variable(sample_date):
    isoformat = sample_date.isoformat()
    result = schema.execute('query Test($date: Date){ date(in: $date) }', variables={'date': sample_date})
    assert not result.errors
    assert result.data == {'date': isoformat}
    result = schema.execute('query Test($date: Date){ date(in: $date) }', variables={'date': isoformat})
    assert not result.errors
    assert result.data == {'date': isoformat}