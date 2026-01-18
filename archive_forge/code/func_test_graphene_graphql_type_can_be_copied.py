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
def test_graphene_graphql_type_can_be_copied():

    class Query(ObjectType):
        field = String()

        def resolve_field(self, info):
            return ''
    schema = Schema(query=Query)
    query_type_copy = copy.copy(schema.graphql_schema.query_type)
    assert query_type_copy.__dict__ == schema.graphql_schema.query_type.__dict__
    assert isinstance(schema.graphql_schema.query_type, GrapheneGraphQLType)