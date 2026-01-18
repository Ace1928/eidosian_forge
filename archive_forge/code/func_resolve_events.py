import datetime
import graphene
from graphene import relay
from graphene.types.resolver import dict_resolver
from ..deduplicator import deflate
def resolve_events(_, info):
    return TEST_DATA['events']