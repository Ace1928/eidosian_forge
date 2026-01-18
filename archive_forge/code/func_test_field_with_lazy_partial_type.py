from functools import partial
from pytest import raises
from ..argument import Argument
from ..field import Field
from ..scalars import String
from ..structures import NonNull
from .utils import MyLazyType
def test_field_with_lazy_partial_type():
    MyType = object()
    field = Field(partial(lambda: MyType))
    assert field.type == MyType