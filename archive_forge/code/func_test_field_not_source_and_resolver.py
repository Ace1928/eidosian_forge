from functools import partial
from pytest import raises
from ..argument import Argument
from ..field import Field
from ..scalars import String
from ..structures import NonNull
from .utils import MyLazyType
def test_field_not_source_and_resolver():
    MyType = object()
    with raises(Exception) as exc_info:
        Field(MyType, source='value', resolver=lambda: None)
    assert str(exc_info.value) == 'A Field cannot have a source and a resolver in at the same time.'