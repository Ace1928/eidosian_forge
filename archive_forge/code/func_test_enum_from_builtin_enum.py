from textwrap import dedent
from ..argument import Argument
from ..enum import Enum, PyEnum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..mutation import Mutation
from ..scalars import String
from ..schema import ObjectType, Schema
def test_enum_from_builtin_enum():
    PyRGB = PyEnum('RGB', 'RED,GREEN,BLUE')
    RGB = Enum.from_enum(PyRGB)
    assert RGB._meta.enum == PyRGB
    assert RGB.RED
    assert RGB.GREEN
    assert RGB.BLUE