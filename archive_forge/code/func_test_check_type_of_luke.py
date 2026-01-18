import pytest
from graphql.error import format_error
from gql import Client, gql
from .schema import StarWarsSchema
def test_check_type_of_luke(client):
    query = gql('\n        query CheckTypeOfLuke {\n          hero(episode: EMPIRE) {\n            __typename\n            name\n          }\n        }\n    ')
    expected = {'hero': {'__typename': 'Human', 'name': 'Luke Skywalker'}}
    result = client.execute(query)
    assert result == expected