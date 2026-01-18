from textwrap import dedent
from ..argument import Argument
from ..enum import Enum, PyEnum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..mutation import Mutation
from ..scalars import String
from ..schema import ObjectType, Schema
def test_enum_can_be_compared():

    class RGB(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3
    assert RGB.RED == 1
    assert RGB.GREEN == 2
    assert RGB.BLUE == 3