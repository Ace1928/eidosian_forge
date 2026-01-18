import re
from textwrap import dedent
from graphql_relay import to_global_id
from ...types import ObjectType, Schema, String
from ..node import Node, is_node
def test_node_requesting_non_node():
    executed = schema.execute('{ node(id:"%s") { __typename } } ' % Node.to_global_id('RootQuery', 1))
    assert executed.errors
    assert re.match('ObjectType .* does not implement the .* interface.', executed.errors[0].message)
    assert executed.data == {'node': None}