import pytest
from graphql import graphql
from graphql.utils.introspection_query import introspection_query
from gql import Client, gql
from .schema import StarWarsSchema
def test_allows_object_fields_in_inline_fragments(client):
    query = '\n        query DroidFieldInFragment {\n          hero {\n            name\n            ... on Droid {\n              primaryFunction\n            }\n          }\n        }\n    '
    assert not validation_errors(client, query)