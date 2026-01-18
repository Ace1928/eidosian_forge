from functools import partial
from pytest import raises
from ..argument import Argument
from ..field import Field
from ..scalars import String
from ..structures import NonNull
from .utils import MyLazyType
def test_field_source():
    MyType = object()
    field = Field(MyType, source='value')
    assert field.resolver(MyInstance(), None) == MyInstance.value