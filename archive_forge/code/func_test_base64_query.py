import base64
from graphql import GraphQLError
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..base64 import Base64
def test_base64_query():
    base64_value = base64.b64encode(b'Random string').decode('utf-8')
    result = schema.execute('{{ base64(input: "{}", match: "Random string") }}'.format(base64_value))
    assert not result.errors
    assert result.data == {'base64': base64_value}