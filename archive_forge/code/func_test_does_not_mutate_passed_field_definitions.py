import copy
from ..argument import Argument
from ..definitions import GrapheneGraphQLType
from ..enum import Enum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import Boolean, Int, String
from ..schema import Schema
from ..structures import List, NonNull
from ..union import Union
def test_does_not_mutate_passed_field_definitions():

    class CommonFields:
        field1 = String()
        field2 = String(id=String())

    class TestObject1(CommonFields, ObjectType):
        pass

    class TestObject2(CommonFields, ObjectType):
        pass
    assert TestObject1._meta.fields == TestObject2._meta.fields

    class CommonFields:
        field1 = String()
        field2 = String()

    class TestInputObject1(CommonFields, InputObjectType):
        pass

    class TestInputObject2(CommonFields, InputObjectType):
        pass
    assert TestInputObject1._meta.fields == TestInputObject2._meta.fields