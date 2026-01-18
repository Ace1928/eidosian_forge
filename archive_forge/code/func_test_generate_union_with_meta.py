from pytest import raises
from ..field import Field
from ..objecttype import ObjectType
from ..union import Union
from ..unmountedtype import UnmountedType
def test_generate_union_with_meta():

    class MyUnion(Union):

        class Meta:
            name = 'MyOtherUnion'
            description = 'Documentation'
            types = (MyObjectType1, MyObjectType2)
    assert MyUnion._meta.name == 'MyOtherUnion'
    assert MyUnion._meta.description == 'Documentation'