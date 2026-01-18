from ..field import Field
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..unmountedtype import UnmountedType
def test_generate_interface_with_fields():

    class MyInterface(Interface):
        field = Field(MyType)
    assert 'field' in MyInterface._meta.fields