from textwrap import dedent
from ..argument import Argument
from ..enum import Enum, PyEnum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..mutation import Mutation
from ..scalars import String
from ..schema import ObjectType, Schema
def test_enum_skip_meta_from_members():

    class RGB1(Enum):

        class Meta:
            name = 'RGB'
        RED = 1
        GREEN = 2
        BLUE = 3
    assert dict(RGB1._meta.enum.__members__) == {'RED': RGB1.RED, 'GREEN': RGB1.GREEN, 'BLUE': RGB1.BLUE}