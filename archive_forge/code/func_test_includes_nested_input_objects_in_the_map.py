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
def test_includes_nested_input_objects_in_the_map():

    class NestedInputObject(InputObjectType):
        value = String()

    class SomeInputObject(InputObjectType):
        nested = InputField(NestedInputObject)

    class SomeMutation(Mutation):
        mutate_something = Field(Article, input=Argument(SomeInputObject))

    class SomeSubscription(Mutation):
        subscribe_to_something = Field(Article, input=Argument(SomeInputObject))
    schema = Schema(query=Query, mutation=SomeMutation, subscription=SomeSubscription)
    type_map = schema.graphql_schema.type_map
    assert type_map['NestedInputObject'].graphene_type is NestedInputObject