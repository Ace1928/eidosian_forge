from pytest import mark
from graphql_relay.utils import base64
from graphene.types import ObjectType, Schema, String
from graphene.relay.connection import Connection, ConnectionField, PageInfo
from graphene.relay.node import Node
def resolve_connection_letters(self, info, **args):
    return LetterConnection(page_info=PageInfo(has_next_page=True, has_previous_page=False), edges=[LetterConnection.Edge(node=Letter(id=0, letter='A'), cursor='a-cursor')])