import pytest
from referencing import Registry, Resource, Specification
import referencing.jsonschema
def test_lookup_recursive_ref_to_bool():
    TRUE = referencing.jsonschema.DRAFT201909.create_resource(True)
    registry = Registry({'http://example.com': TRUE})
    resolved = referencing.jsonschema.lookup_recursive_ref(resolver=registry.resolver(base_uri='http://example.com'))
    assert resolved.contents == TRUE.contents