import datetime
import pytz
from graphql import GraphQLError
from pytest import fixture
from ..datetime import Date, DateTime, Time
from ..objecttype import ObjectType
from ..schema import Schema
@fixture
def sample_time(sample_datetime):
    time = datetime.time(sample_datetime.hour, sample_datetime.minute, sample_datetime.second, sample_datetime.microsecond, sample_datetime.tzinfo)
    return time