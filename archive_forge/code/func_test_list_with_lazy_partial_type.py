from functools import partial
from pytest import raises
from ..scalars import String
from ..structures import List, NonNull
from .utils import MyLazyType
def test_list_with_lazy_partial_type():
    MyType = object()
    field = List(partial(lambda: MyType))
    assert field.of_type == MyType