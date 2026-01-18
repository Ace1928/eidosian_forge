from ..field import Field
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..unmountedtype import UnmountedType
def test_generate_interface_inherit_abstracttype_reversed():

    class MyAbstractType:
        field1 = MyScalar()

    class MyInterface(MyAbstractType, Interface):
        field2 = MyScalar()
    assert list(MyInterface._meta.fields) == ['field1', 'field2']
    assert [type(x) for x in MyInterface._meta.fields.values()] == [Field, Field]