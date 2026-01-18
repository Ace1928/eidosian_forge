from textwrap import dedent
from ..argument import Argument
from ..enum import Enum, PyEnum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..mutation import Mutation
from ..scalars import String
from ..schema import ObjectType, Schema
def test_enum_iteration():

    class TestEnum(Enum):
        FIRST = 1
        SECOND = 2
    result = []
    expected_values = ['FIRST', 'SECOND']
    for c in TestEnum:
        result.append(c.name)
    assert result == expected_values