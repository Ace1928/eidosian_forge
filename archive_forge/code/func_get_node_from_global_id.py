from textwrap import dedent
from graphql import graphql_sync
from ...types import Interface, ObjectType, Schema
from ...types.scalars import Int, String
from ..node import Node
@staticmethod
def get_node_from_global_id(info, id, only_type=None):
    assert info.schema is graphql_schema
    if id in user_data:
        return user_data.get(id)
    else:
        return photo_data.get(id)