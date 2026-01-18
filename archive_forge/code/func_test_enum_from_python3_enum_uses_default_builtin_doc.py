from textwrap import dedent
from ..argument import Argument
from ..enum import Enum, PyEnum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..mutation import Mutation
from ..scalars import String
from ..schema import ObjectType, Schema
def test_enum_from_python3_enum_uses_default_builtin_doc():
    RGB = Enum('RGB', 'RED,GREEN,BLUE')
    assert RGB._meta.description == 'An enumeration.'