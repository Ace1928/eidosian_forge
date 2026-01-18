from textwrap import dedent
from ..argument import Argument
from ..enum import Enum, PyEnum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..mutation import Mutation
from ..scalars import String
from ..schema import ObjectType, Schema
def test_enum_value_as_unmounted_field():

    class RGB(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3
    unmounted = RGB()
    unmounted_field = unmounted.Field()
    assert isinstance(unmounted_field, Field)
    assert unmounted_field.type == RGB