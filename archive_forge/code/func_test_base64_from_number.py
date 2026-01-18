import base64
from graphql import GraphQLError
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..base64 import Base64
def test_base64_from_number():
    base64_value = base64.b64encode(b'42').decode('utf-8')
    result = schema.execute('{ numberAsBase64 }')
    assert not result.errors
    assert result.data == {'numberAsBase64': base64_value}