import pytest
from gql import Client
from gql.dsl import DSLSchema
from .schema import StarWarsSchema
def test_hero_name_query_result(ds):
    result = ds.query(ds.Query.hero.select(ds.Character.name))
    expected = {'hero': {'name': 'R2-D2'}}
    assert result == expected