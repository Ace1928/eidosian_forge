from functools import partial
from pytest import raises
from ..scalars import String
from ..structures import List, NonNull
from .utils import MyLazyType
def test_list_inherited_works_nonnull():
    _list = List(NonNull(String))
    assert isinstance(_list.of_type, NonNull)
    assert _list.of_type.of_type == String