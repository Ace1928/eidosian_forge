import datetime
import graphene
from graphene import relay
from graphene.types.resolver import dict_resolver
from ..deduplicator import deflate
def test_does_not_modify_first_instance_of_an_object_nested():
    response = {'data': [{'__typename': 'foo', 'bar1': {'__typename': 'bar', 'id': 1, 'name': 'bar'}, 'bar2': {'__typename': 'bar', 'id': 1, 'name': 'bar'}, 'id': 1}, {'__typename': 'foo', 'bar1': {'__typename': 'bar', 'id': 1, 'name': 'bar'}, 'bar2': {'__typename': 'bar', 'id': 1, 'name': 'bar'}, 'id': 2}]}
    deflated_response = deflate(response)
    assert deflated_response == {'data': [{'__typename': 'foo', 'bar1': {'__typename': 'bar', 'id': 1, 'name': 'bar'}, 'bar2': {'__typename': 'bar', 'id': 1, 'name': 'bar'}, 'id': 1}, {'__typename': 'foo', 'bar1': {'__typename': 'bar', 'id': 1}, 'bar2': {'__typename': 'bar', 'id': 1}, 'id': 2}]}