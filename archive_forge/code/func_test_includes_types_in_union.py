import copy
from ..argument import Argument
from ..definitions import GrapheneGraphQLType
from ..enum import Enum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import Boolean, Int, String
from ..schema import Schema
from ..structures import List, NonNull
from ..union import Union
def test_includes_types_in_union():

    class SomeType(ObjectType):
        a = String()

    class OtherType(ObjectType):
        b = String()

    class MyUnion(Union):

        class Meta:
            types = (SomeType, OtherType)

    class Query(ObjectType):
        union = Field(MyUnion)
    schema = Schema(query=Query)
    type_map = schema.graphql_schema.type_map
    assert type_map['OtherType'].graphene_type is OtherType
    assert type_map['SomeType'].graphene_type is SomeType