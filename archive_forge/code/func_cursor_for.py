from pytest import mark
from graphql_relay.utils import base64
from graphene.types import ObjectType, Schema, String
from graphene.relay.connection import Connection, ConnectionField, PageInfo
from graphene.relay.node import Node
def cursor_for(ltr):
    letter = letters[ltr]
    return base64('arrayconnection:%s' % letter.id)