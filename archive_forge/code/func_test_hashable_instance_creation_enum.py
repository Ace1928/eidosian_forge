from textwrap import dedent
from ..argument import Argument
from ..enum import Enum, PyEnum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..mutation import Mutation
from ..scalars import String
from ..schema import ObjectType, Schema
def test_hashable_instance_creation_enum():
    Episode = Enum('Episode', [('NEWHOPE', 4), ('EMPIRE', 5), ('JEDI', 6)])
    trilogy_map = {Episode.NEWHOPE: 'better', Episode.EMPIRE: 'best', 5: 'foo'}
    assert trilogy_map[Episode.NEWHOPE] == 'better'
    assert trilogy_map[Episode.EMPIRE] == 'best'
    assert trilogy_map[5] == 'foo'