import base64
from graphql import GraphQLError
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..base64 import Base64
def resolve_base64(self, info, _in=None, _match=None):
    if _match:
        assert _in == _match
    return _in