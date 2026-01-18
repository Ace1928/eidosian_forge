from ..field import Field
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..unmountedtype import UnmountedType
def test_generate_interface_unmountedtype():

    class MyInterface(Interface):
        field = MyScalar()
    assert 'field' in MyInterface._meta.fields
    assert isinstance(MyInterface._meta.fields['field'], Field)