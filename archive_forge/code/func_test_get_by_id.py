import re
from uuid import uuid4
from graphql import graphql_sync
from ..id_type import BaseGlobalIDType, SimpleGlobalIDType, UUIDGlobalIDType
from ..node import Node
from ...types import Int, ObjectType, Schema, String
def test_get_by_id(self):
    query = 'query {\n            user(id: 2) {\n                id\n                name\n            }\n        }'
    result = graphql_sync(self.graphql_schema, query)
    assert not result.errors
    assert result.data['user']['id'] == self.user_list[1]['id']
    assert result.data['user']['name'] == self.user_list[1]['name']