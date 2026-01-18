from textwrap import dedent
from ..argument import Argument
from ..enum import Enum, PyEnum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..mutation import Mutation
from ..scalars import String
from ..schema import ObjectType, Schema
def test_enum_types():
    from enum import Enum as PyEnum

    class Color(PyEnum):
        """Primary colors"""
        RED = 1
        YELLOW = 2
        BLUE = 3
    GColor = Enum.from_enum(Color)

    class Query(ObjectType):
        color = GColor(required=True)

        def resolve_color(_, info):
            return Color.RED
    schema = Schema(query=Query)
    assert str(schema).strip() == dedent('\n            type Query {\n              color: Color!\n            }\n\n            """Primary colors"""\n            enum Color {\n              RED\n              YELLOW\n              BLUE\n            }\n            ').strip()