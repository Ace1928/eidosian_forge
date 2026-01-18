from pytest import raises
from ..field import Field
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..structures import NonNull
from ..unmountedtype import UnmountedType
def test_generate_objecttype_with_private_attributes():

    class MyObjectType(ObjectType):

        def __init__(self, _private_state=None, **kwargs):
            self._private_state = _private_state
            super().__init__(**kwargs)
        _private_state = None
    assert '_private_state' not in MyObjectType._meta.fields
    assert hasattr(MyObjectType, '_private_state')
    m = MyObjectType(_private_state='custom')
    assert m._private_state == 'custom'
    with raises(TypeError):
        MyObjectType(_other_private_state='Wrong')