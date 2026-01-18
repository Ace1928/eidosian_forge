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
def test_includes_interfaces_subtypes_in_the_type_map():

    class SomeInterface(Interface):
        f = Int()

    class SomeSubtype(ObjectType):

        class Meta:
            interfaces = (SomeInterface,)

    class Query(ObjectType):
        iface = Field(SomeInterface)
    schema = Schema(query=Query, types=[SomeSubtype])
    type_map = schema.graphql_schema.type_map
    assert type_map['SomeSubtype'].graphene_type is SomeSubtype