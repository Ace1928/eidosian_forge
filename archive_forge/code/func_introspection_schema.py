import pytest
from graphql import graphql
from graphql.utils.introspection_query import introspection_query
from gql import Client, gql
from .schema import StarWarsSchema
@pytest.fixture
def introspection_schema():
    return Client(introspection=introspection)