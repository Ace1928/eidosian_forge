import pytest
from referencing import Registry, Resource, Specification
import referencing.jsonschema
def test_lookup_trivial_dynamic_ref():
    one = referencing.jsonschema.DRAFT202012.create_resource({'$dynamicAnchor': 'foo'})
    resolver = Registry().with_resource('http://example.com', one).resolver()
    resolved = resolver.lookup('http://example.com#foo')
    assert resolved.contents == one.contents