from pytest import raises
from ..field import Field
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..structures import NonNull
from ..unmountedtype import UnmountedType
def test_generate_objecttype_unmountedtype():

    class MyObjectType(ObjectType):
        field = MyScalar()
    assert 'field' in MyObjectType._meta.fields
    assert isinstance(MyObjectType._meta.fields['field'], Field)