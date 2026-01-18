import pytest
from graphql import graphql
from graphql.utils.introspection_query import introspection_query
from gql import Client, gql
from .schema import StarWarsSchema
def test_allows_object_fields_in_fragments(client):
    query = '\n        query DroidFieldInFragment {\n          hero {\n            name\n            ...DroidFields\n          }\n        }\n        fragment DroidFields on Droid {\n          primaryFunction\n        }\n    '
    assert not validation_errors(client, query)