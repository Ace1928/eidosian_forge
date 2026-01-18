from functools import partial
from pytest import raises
from ..argument import Argument
from ..field import Field
from ..scalars import String
from ..structures import NonNull
from .utils import MyLazyType
def test_field_with_string_type():
    field = Field('graphene.types.tests.utils.MyLazyType')
    assert field.type == MyLazyType