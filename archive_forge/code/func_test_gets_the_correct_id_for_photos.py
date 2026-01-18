from textwrap import dedent
from graphql import graphql_sync
from ...types import Interface, ObjectType, Schema
from ...types.scalars import Int, String
from ..node import Node
def test_gets_the_correct_id_for_photos():
    query = '\n      {\n        node(id: "4") {\n          id\n        }\n      }\n    '
    expected = {'node': {'id': '4'}}
    result = graphql_sync(graphql_schema, query)
    assert not result.errors
    assert result.data == expected