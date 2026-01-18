import pytest
from referencing import Registry, Resource, Specification
import referencing.jsonschema
def test_multiple_lookup_dynamic_ref_to_nondynamic_ref():
    one = referencing.jsonschema.DRAFT202012.create_resource({'$anchor': 'fooAnchor'})
    two = referencing.jsonschema.DRAFT202012.create_resource({'$id': 'http://example.com', '$dynamicAnchor': 'fooAnchor', '$defs': {'foo': {'$id': 'foo', '$dynamicAnchor': 'fooAnchor', '$defs': {'bar': True, 'baz': {'$dynamicAnchor': 'fooAnchor'}}}}})
    resolver = Registry().with_resources([('http://example.com', two), ('http://example.com/foo/', one), ('http://example.com/foo/bar', two)]).resolver()
    first = resolver.lookup('http://example.com')
    second = first.resolver.lookup('foo/')
    resolver = second.resolver.lookup('bar').resolver
    fourth = resolver.lookup('#fooAnchor')
    assert fourth.contents == two.contents