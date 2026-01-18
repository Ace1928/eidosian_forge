from textwrap import dedent
from ..argument import Argument
from ..enum import Enum, PyEnum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..mutation import Mutation
from ..scalars import String
from ..schema import ObjectType, Schema
def test_enum_resolver():
    from enum import Enum as PyEnum

    class Color(PyEnum):
        RED = 1
        GREEN = 2
        BLUE = 3
    GColor = Enum.from_enum(Color)

    class Query(ObjectType):
        color = GColor(required=True)

        def resolve_color(_, info):
            return Color.RED
    schema = Schema(query=Query)
    results = schema.execute('query { color }')
    assert not results.errors
    assert results.data['color'] == Color.RED.name