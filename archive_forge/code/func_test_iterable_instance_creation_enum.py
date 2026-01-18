from textwrap import dedent
from ..argument import Argument
from ..enum import Enum, PyEnum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..mutation import Mutation
from ..scalars import String
from ..schema import ObjectType, Schema
def test_iterable_instance_creation_enum():
    TestEnum = Enum('TestEnum', [('FIRST', 1), ('SECOND', 2)])
    result = []
    expected_values = ['FIRST', 'SECOND']
    for c in TestEnum:
        result.append(c.name)
    assert result == expected_values