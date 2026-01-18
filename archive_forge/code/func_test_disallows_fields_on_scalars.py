import pytest
from graphql import graphql
from graphql.utils.introspection_query import introspection_query
from gql import Client, gql
from .schema import StarWarsSchema
def test_disallows_fields_on_scalars(client):
    query = '\n        query HeroFieldsOnScalarQuery {\n          hero {\n            name {\n              firstCharacterOfName\n            }\n          }\n        }\n    '
    assert validation_errors(client, query)