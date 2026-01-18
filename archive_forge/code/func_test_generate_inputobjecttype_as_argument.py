from graphql import Undefined
from ..argument import Argument
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..objecttype import ObjectType
from ..scalars import Boolean, String
from ..schema import Schema
from ..unmountedtype import UnmountedType
from ... import NonNull
def test_generate_inputobjecttype_as_argument():

    class MyInputObjectType(InputObjectType):
        field = MyScalar()

    class MyObjectType(ObjectType):
        field = Field(MyType, input=MyInputObjectType())
    assert 'field' in MyObjectType._meta.fields
    field = MyObjectType._meta.fields['field']
    assert isinstance(field, Field)
    assert field.type == MyType
    assert 'input' in field.args
    assert isinstance(field.args['input'], Argument)
    assert field.args['input'].type == MyInputObjectType