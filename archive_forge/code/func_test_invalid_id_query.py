import pytest
from graphql.error import format_error
from gql import Client, gql
from .schema import StarWarsSchema
def test_invalid_id_query(client):
    query = gql('\n        query humanQuery($id: String!) {\n          human(id: $id) {\n            name\n          }\n        }\n    ')
    params = {'id': 'not a valid id'}
    expected = {'human': None}
    result = client.execute(query, variable_values=params)
    assert result == expected