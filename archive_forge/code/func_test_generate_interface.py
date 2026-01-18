from ..field import Field
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..unmountedtype import UnmountedType
def test_generate_interface():

    class MyInterface(Interface):
        """Documentation"""
    assert MyInterface._meta.name == 'MyInterface'
    assert MyInterface._meta.description == 'Documentation'
    assert MyInterface._meta.fields == {}