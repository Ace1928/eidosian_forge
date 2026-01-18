import re
from textwrap import dedent
from graphql_relay import to_global_id
from ...types import ObjectType, Schema, String
from ..node import Node, is_node
def test_node_field_only_type_wrong():
    executed = schema.execute('{ onlyNode(id:"%s") { __typename, name } } ' % Node.to_global_id('MyOtherNode', 1))
    assert len(executed.errors) == 1
    assert str(executed.errors[0]).startswith('Must receive a MyNode id.')
    assert executed.data == {'onlyNode': None}