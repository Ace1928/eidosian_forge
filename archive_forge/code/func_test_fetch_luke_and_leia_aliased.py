import pytest
from graphql.error import format_error
from gql import Client, gql
from .schema import StarWarsSchema
def test_fetch_luke_and_leia_aliased(client):
    query = gql('\n        query FetchLukeAndLeiaAliased {\n          luke: human(id: "1000") {\n            name\n          }\n          leia: human(id: "1003") {\n            name\n          }\n        }\n    ')
    expected = {'luke': {'name': 'Luke Skywalker'}, 'leia': {'name': 'Leia Organa'}}
    result = client.execute(query)
    assert result == expected