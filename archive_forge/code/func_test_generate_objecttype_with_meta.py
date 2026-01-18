from pytest import raises
from ..field import Field
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..structures import NonNull
from ..unmountedtype import UnmountedType
def test_generate_objecttype_with_meta():

    class MyObjectType(ObjectType):

        class Meta:
            name = 'MyOtherObjectType'
            description = 'Documentation'
            interfaces = (MyType,)
    assert MyObjectType._meta.name == 'MyOtherObjectType'
    assert MyObjectType._meta.description == 'Documentation'
    assert MyObjectType._meta.interfaces == (MyType,)