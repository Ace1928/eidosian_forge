import pytest
from graphql.error import format_error
from gql import Client, gql
from .schema import StarWarsSchema
def test_fetch_luke_aliased(client):
    query = gql('\n        query FetchLukeAliased {\n          luke: human(id: "1000") {\n            name\n          }\n        }\n    ')
    expected = {'luke': {'name': 'Luke Skywalker'}}
    result = client.execute(query)
    assert result == expected