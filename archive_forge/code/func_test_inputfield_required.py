from functools import partial
from pytest import raises
from ..inputfield import InputField
from ..structures import NonNull
from .utils import MyLazyType
def test_inputfield_required():
    MyType = object()
    field = InputField(MyType, required=True)
    assert isinstance(field.type, NonNull)
    assert field.type.of_type == MyType