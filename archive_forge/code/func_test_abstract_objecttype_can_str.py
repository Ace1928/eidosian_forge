from pytest import raises
from ..field import Field
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..structures import NonNull
from ..unmountedtype import UnmountedType
def test_abstract_objecttype_can_str():

    class MyObjectType(ObjectType):

        class Meta:
            abstract = True
        field = MyScalar()
    assert str(MyObjectType) == 'MyObjectType'