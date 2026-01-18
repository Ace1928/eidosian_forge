from textwrap import dedent
from ..argument import Argument
from ..enum import Enum, PyEnum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..mutation import Mutation
from ..scalars import String
from ..schema import ObjectType, Schema
def custom_description(value):
    if not value:
        return 'StarWars Episodes'
    return 'New Hope Episode' if value == Episode.NEWHOPE else 'Other'