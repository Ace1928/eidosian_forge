import pytest
from graphql.error import format_error
from gql import Client, gql
from .schema import StarWarsSchema
def test_duplicate_fields(client):
    query = gql('\n        query DuplicateFields {\n          luke: human(id: "1000") {\n            name\n            homePlanet\n          }\n          leia: human(id: "1003") {\n            name\n            homePlanet\n          }\n        }\n    ')
    expected = {'luke': {'name': 'Luke Skywalker', 'homePlanet': 'Tatooine'}, 'leia': {'name': 'Leia Organa', 'homePlanet': 'Alderaan'}}
    result = client.execute(query)
    assert result == expected