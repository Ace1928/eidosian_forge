from textwrap import dedent
from ..argument import Argument
from ..enum import Enum, PyEnum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..mutation import Mutation
from ..scalars import String
from ..schema import ObjectType, Schema
def test_enum_description_member_not_interpreted_as_property():

    class RGB(Enum):
        """Description"""
        red = 'red'
        green = 'green'
        blue = 'blue'
        description = 'description'
        deprecation_reason = 'deprecation_reason'

    class Query(ObjectType):
        color = RGB()

        def resolve_color(_, info):
            return RGB.description
    values = RGB._meta.enum.__members__.values()
    assert sorted((v.name for v in values)) == ['blue', 'deprecation_reason', 'description', 'green', 'red']
    schema = Schema(query=Query)
    results = schema.execute('query { color }')
    assert not results.errors
    assert results.data['color'] == RGB.description.name