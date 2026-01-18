from functools import partial
from pytest import raises
from ..inputfield import InputField
from ..structures import NonNull
from .utils import MyLazyType
def test_inputfield_with_string_type():
    field = InputField('graphene.types.tests.utils.MyLazyType')
    assert field.type == MyLazyType