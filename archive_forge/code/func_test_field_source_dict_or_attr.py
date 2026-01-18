from functools import partial
from pytest import raises
from ..argument import Argument
from ..field import Field
from ..scalars import String
from ..structures import NonNull
from .utils import MyLazyType
def test_field_source_dict_or_attr():
    MyType = object()
    field = Field(MyType, source='value')
    assert field.resolver(MyInstance(), None) == MyInstance.value
    assert field.resolver({'value': MyInstance.value}, None) == MyInstance.value