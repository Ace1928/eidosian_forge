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
def test_generate_inputobjecttype_with_meta():

    class MyInputObjectType(InputObjectType):

        class Meta:
            name = 'MyOtherInputObjectType'
            description = 'Documentation'
    assert MyInputObjectType._meta.name == 'MyOtherInputObjectType'
    assert MyInputObjectType._meta.description == 'Documentation'