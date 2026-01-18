from graphql import parse, validate
from ...types import Schema, ObjectType, String
from ..disable_introspection import DisableIntrospection
def test_disallows_introspection_queries():
    errors = run_query('{ __schema { queryType { name } } }')
    assert len(errors) == 1
    assert errors[0].message == "Cannot query '__schema': introspection is disabled."