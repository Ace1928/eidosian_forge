import base64
from graphql import GraphQLError
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..base64 import Base64
def test_base64_query_none():
    result = schema.execute('{ base64 }')
    assert not result.errors
    assert result.data == {'base64': None}