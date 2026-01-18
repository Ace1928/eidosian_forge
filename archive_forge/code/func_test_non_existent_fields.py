import pytest
from graphql import graphql
from graphql.utils.introspection_query import introspection_query
from gql import Client, gql
from .schema import StarWarsSchema
def test_non_existent_fields(client):
    query = '\n        query HeroSpaceshipQuery {\n          hero {\n            favoriteSpaceship\n          }\n        }\n    '
    assert validation_errors(client, query)