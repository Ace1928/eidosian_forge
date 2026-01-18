import base64
from graphql import GraphQLError
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..base64 import Base64
def resolve_string_as_base64(self, info):
    return 'Spam and eggs'