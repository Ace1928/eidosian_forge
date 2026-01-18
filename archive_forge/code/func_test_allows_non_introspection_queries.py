from graphql import parse, validate
from ...types import Schema, ObjectType, String
from ..disable_introspection import DisableIntrospection
def test_allows_non_introspection_queries():
    errors = run_query('{ name }')
    assert len(errors) == 0