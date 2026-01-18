import re
from uuid import uuid4
from graphql import graphql_sync
from ..id_type import BaseGlobalIDType, SimpleGlobalIDType, UUIDGlobalIDType
from ..node import Node
from ...types import Int, ObjectType, Schema, String
@classmethod
def resolve_global_id(cls, info, global_id):
    _type = info.return_type.graphene_type._meta.name
    return (_type, global_id)