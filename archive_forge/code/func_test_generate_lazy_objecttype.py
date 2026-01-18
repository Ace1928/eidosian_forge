from pytest import raises
from ..field import Field
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import String
from ..schema import Schema
from ..structures import NonNull
from ..unmountedtype import UnmountedType
def test_generate_lazy_objecttype():

    class MyObjectType(ObjectType):
        example = Field(lambda: InnerObjectType, required=True)

    class InnerObjectType(ObjectType):
        field = Field(MyType)
    assert MyObjectType._meta.name == 'MyObjectType'
    example_field = MyObjectType._meta.fields['example']
    assert isinstance(example_field.type, NonNull)
    assert example_field.type.of_type == InnerObjectType