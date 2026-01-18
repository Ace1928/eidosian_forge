import re
from textwrap import dedent
from graphql_relay import to_global_id
from ...types import ObjectType, Schema, String
from ..node import Node, is_node
def test_node_field_only_type():
    executed = schema.execute('{ onlyNode(id:"%s") { __typename, name } } ' % Node.to_global_id('MyNode', 1))
    assert not executed.errors
    assert executed.data == {'onlyNode': {'__typename': 'MyNode', 'name': '1'}}