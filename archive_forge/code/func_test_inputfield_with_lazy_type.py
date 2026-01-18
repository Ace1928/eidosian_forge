from functools import partial
from pytest import raises
from ..inputfield import InputField
from ..structures import NonNull
from .utils import MyLazyType
def test_inputfield_with_lazy_type():
    MyType = object()
    field = InputField(lambda: MyType)
    assert field.type == MyType