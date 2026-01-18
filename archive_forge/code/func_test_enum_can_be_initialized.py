from textwrap import dedent
from ..argument import Argument
from ..enum import Enum, PyEnum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..mutation import Mutation
from ..scalars import String
from ..schema import ObjectType, Schema
def test_enum_can_be_initialized():

    class RGB(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3
    assert RGB.get(1) == RGB.RED
    assert RGB.get(2) == RGB.GREEN
    assert RGB.get(3) == RGB.BLUE