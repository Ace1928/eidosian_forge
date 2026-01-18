import pytest
from referencing import Registry, Resource, Specification
import referencing.jsonschema
def test_multiple_lookup_trivial_dynamic_ref():
    TRUE = referencing.jsonschema.DRAFT202012.create_resource(True)
    root = referencing.jsonschema.DRAFT202012.create_resource({'$id': 'http://example.com', '$dynamicAnchor': 'fooAnchor', '$defs': {'foo': {'$id': 'foo', '$dynamicAnchor': 'fooAnchor', '$defs': {'bar': True, 'baz': {'$dynamicAnchor': 'fooAnchor'}}}}})
    resolver = Registry().with_resources([('http://example.com', root), ('http://example.com/foo/', TRUE), ('http://example.com/foo/bar', root)]).resolver()
    first = resolver.lookup('http://example.com')
    second = first.resolver.lookup('foo/')
    resolver = second.resolver.lookup('bar').resolver
    fourth = resolver.lookup('#fooAnchor')
    assert fourth.contents == root.contents