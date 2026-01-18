import pytest
from graphql.error import format_error
from gql import Client, gql
from .schema import StarWarsSchema
def test_nested_query(client):
    query = gql('\n        query NestedQuery {\n          hero {\n            name\n            friends {\n              name\n              appearsIn\n              friends {\n                name\n              }\n            }\n          }\n        }\n    ')
    expected = {'hero': {'name': 'R2-D2', 'friends': [{'name': 'Luke Skywalker', 'appearsIn': ['NEWHOPE', 'EMPIRE', 'JEDI'], 'friends': [{'name': 'Han Solo'}, {'name': 'Leia Organa'}, {'name': 'C-3PO'}, {'name': 'R2-D2'}]}, {'name': 'Han Solo', 'appearsIn': ['NEWHOPE', 'EMPIRE', 'JEDI'], 'friends': [{'name': 'Luke Skywalker'}, {'name': 'Leia Organa'}, {'name': 'R2-D2'}]}, {'name': 'Leia Organa', 'appearsIn': ['NEWHOPE', 'EMPIRE', 'JEDI'], 'friends': [{'name': 'Luke Skywalker'}, {'name': 'Han Solo'}, {'name': 'C-3PO'}, {'name': 'R2-D2'}]}]}}
    result = client.execute(query)
    assert result == expected