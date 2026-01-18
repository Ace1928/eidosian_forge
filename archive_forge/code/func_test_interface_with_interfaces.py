from graphql import Undefined
from graphql.type import (
from ..dynamic import Dynamic
from ..enum import Enum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import Int, String
from ..schema import Schema
from ..structures import List, NonNull
def test_interface_with_interfaces():

    class FooInterface(Interface):
        foo = String()

    class BarInterface(Interface):

        class Meta:
            interfaces = [FooInterface]
        foo = String()
        bar = String()
    type_map = create_type_map([FooInterface, BarInterface])
    assert 'FooInterface' in type_map
    foo_graphql_type = type_map['FooInterface']
    assert isinstance(foo_graphql_type, GraphQLInterfaceType)
    assert foo_graphql_type.name == 'FooInterface'
    assert 'BarInterface' in type_map
    bar_graphql_type = type_map['BarInterface']
    assert isinstance(bar_graphql_type, GraphQLInterfaceType)
    assert bar_graphql_type.name == 'BarInterface'
    fields = bar_graphql_type.fields
    assert list(fields) == ['foo', 'bar']
    assert isinstance(fields['foo'], GraphQLField)
    assert isinstance(fields['bar'], GraphQLField)
    assert list(bar_graphql_type.interfaces) == list([foo_graphql_type])