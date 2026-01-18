from pytest import raises
from ..field import Field
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..structures import NonNull
from ..unmountedtype import UnmountedType
def test_generate_objecttype_inherit_abstracttype():

    class MyAbstractType:
        field1 = MyScalar()

    class MyObjectType(ObjectType, MyAbstractType):
        field2 = MyScalar()
    assert MyObjectType._meta.description is None
    assert MyObjectType._meta.interfaces == ()
    assert MyObjectType._meta.name == 'MyObjectType'
    assert list(MyObjectType._meta.fields) == ['field1', 'field2']
    assert list(map(type, MyObjectType._meta.fields.values())) == [Field, Field]