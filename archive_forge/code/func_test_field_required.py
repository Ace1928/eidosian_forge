from functools import partial
from pytest import raises
from ..argument import Argument
from ..field import Field
from ..scalars import String
from ..structures import NonNull
from .utils import MyLazyType
def test_field_required():
    MyType = object()
    field = Field(MyType, required=True)
    assert isinstance(field.type, NonNull)
    assert field.type.of_type == MyType