from pytest import raises
from ..field import Field
from ..objecttype import ObjectType
from ..union import Union
from ..unmountedtype import UnmountedType
def test_generate_union_with_no_types():
    with raises(Exception) as exc_info:

        class MyUnion(Union):
            pass
    assert str(exc_info.value) == 'Must provide types for Union MyUnion.'