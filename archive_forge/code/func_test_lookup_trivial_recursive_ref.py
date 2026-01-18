import pytest
from referencing import Registry, Resource, Specification
import referencing.jsonschema
def test_lookup_trivial_recursive_ref():
    one = referencing.jsonschema.DRAFT201909.create_resource({'$recursiveAnchor': True})
    resolver = Registry().with_resource('http://example.com', one).resolver()
    first = resolver.lookup('http://example.com')
    resolved = referencing.jsonschema.lookup_recursive_ref(resolver=first.resolver)
    assert resolved.contents == one.contents