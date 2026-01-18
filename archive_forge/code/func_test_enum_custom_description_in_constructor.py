from textwrap import dedent
from ..argument import Argument
from ..enum import Enum, PyEnum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..mutation import Mutation
from ..scalars import String
from ..schema import ObjectType, Schema
def test_enum_custom_description_in_constructor():
    description = 'An enumeration, but with a custom description'
    RGB = Enum('RGB', 'RED,GREEN,BLUE', description=description)
    assert RGB._meta.description == description