from textwrap import dedent
from graphql import graphql_sync
from ...types import Interface, ObjectType, Schema
from ...types.scalars import Int, String
from ..node import Node
def test_ignores_photo_fragments_on_user():
    query = '\n      {\n        node(id: "1") {\n          id\n          ... on Photo {\n            width\n          }\n        }\n      }\n    '
    expected = {'node': {'id': '1'}}
    result = graphql_sync(graphql_schema, query)
    assert not result.errors
    assert result.data == expected