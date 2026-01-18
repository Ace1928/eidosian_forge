from textwrap import dedent
from ..argument import Argument
from ..enum import Enum, PyEnum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..mutation import Mutation
from ..scalars import String
from ..schema import ObjectType, Schema
def test_mutation_enum_input():

    class RGB(Enum):
        """Available colors"""
        RED = 1
        GREEN = 2
        BLUE = 3
    color_input = None

    class CreatePaint(Mutation):

        class Arguments:
            color = RGB(required=True)
        color = RGB(required=True)

        def mutate(_, info, color):
            nonlocal color_input
            color_input = color
            return CreatePaint(color=color)

    class MyMutation(ObjectType):
        create_paint = CreatePaint.Field()

    class Query(ObjectType):
        a = String()
    schema = Schema(query=Query, mutation=MyMutation)
    result = schema.execute(' mutation MyMutation {\n        createPaint(color: RED) {\n            color\n        }\n    }\n    ')
    assert not result.errors
    assert result.data == {'createPaint': {'color': 'RED'}}
    assert color_input == RGB.RED