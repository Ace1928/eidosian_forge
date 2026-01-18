from textwrap import dedent
from ..argument import Argument
from ..enum import Enum, PyEnum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..mutation import Mutation
from ..scalars import String
from ..schema import ObjectType, Schema
def test_enum_construction():

    class RGB(Enum):
        """Description"""
        RED = 1
        GREEN = 2
        BLUE = 3

        @property
        def description(self):
            return f'Description {self.name}'
    assert RGB._meta.name == 'RGB'
    assert RGB._meta.description == 'Description'
    values = RGB._meta.enum.__members__.values()
    assert sorted((v.name for v in values)) == ['BLUE', 'GREEN', 'RED']
    assert sorted((v.description for v in values)) == ['Description BLUE', 'Description GREEN', 'Description RED']