import base64
from graphql import GraphQLError
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..base64 import Base64
def test_base64_query_with_variable():
    base64_value = base64.b64encode(b'Another string').decode('utf-8')
    result = schema.execute('\n        query GetBase64($base64: Base64) {\n            base64(input: $base64, match: "Another string")\n        }\n        ', variables={'base64': base64_value})
    assert not result.errors
    assert result.data == {'base64': base64_value}