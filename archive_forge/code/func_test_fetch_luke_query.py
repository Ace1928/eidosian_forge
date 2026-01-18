import pytest
from graphql.error import format_error
from gql import Client, gql
from .schema import StarWarsSchema
def test_fetch_luke_query(client):
    query = gql('\n        query FetchLukeQuery {\n          human(id: "1000") {\n            name\n          }\n        }\n    ')
    expected = {'human': {'name': 'Luke Skywalker'}}
    result = client.execute(query)
    assert result == expected