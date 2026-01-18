import re
from textwrap import dedent
from graphql_relay import to_global_id
from ...types import ObjectType, Schema, String
from ..node import Node, is_node
def test_node_requesting_unknown_type():
    executed = schema.execute('{ node(id:"%s") { __typename } } ' % Node.to_global_id('UnknownType', 1))
    assert executed.errors
    assert re.match('Relay Node .* not found in schema', executed.errors[0].message)
    assert executed.data == {'node': None}