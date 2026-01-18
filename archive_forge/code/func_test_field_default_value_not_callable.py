from functools import partial
from pytest import raises
from ..argument import Argument
from ..field import Field
from ..scalars import String
from ..structures import NonNull
from .utils import MyLazyType
def test_field_default_value_not_callable():
    MyType = object()
    try:
        Field(MyType, default_value=lambda: True)
    except AssertionError as e:
        assert 'The default value can not be a function but received' in str(e)