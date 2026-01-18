from textwrap import dedent
from graphql import graphql_sync
from ...types import Interface, ObjectType, Schema
from ...types.scalars import Int, String
from ..node import Node
def test_have_correct_node_interface():
    query = '\n      {\n        __type(name: "Node") {\n          name\n          kind\n          fields {\n            name\n            type {\n              kind\n              ofType {\n                name\n                kind\n              }\n            }\n          }\n        }\n      }\n    '
    expected = {'__type': {'name': 'Node', 'kind': 'INTERFACE', 'fields': [{'name': 'id', 'type': {'kind': 'NON_NULL', 'ofType': {'name': 'ID', 'kind': 'SCALAR'}}}]}}
    result = graphql_sync(graphql_schema, query)
    assert not result.errors
    assert result.data == expected