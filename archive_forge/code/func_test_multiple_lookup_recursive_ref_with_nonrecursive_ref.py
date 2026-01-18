import pytest
from referencing import Registry, Resource, Specification
import referencing.jsonschema
def test_multiple_lookup_recursive_ref_with_nonrecursive_ref():
    one = referencing.jsonschema.DRAFT201909.create_resource({'$recursiveAnchor': True})
    two = referencing.jsonschema.DRAFT201909.create_resource({'$id': 'http://example.com', '$recursiveAnchor': True, '$defs': {'foo': {'$id': 'foo', '$recursiveAnchor': True, '$defs': {'bar': True, 'baz': {'$recursiveAnchor': True, '$anchor': 'fooAnchor'}}}}})
    three = referencing.jsonschema.DRAFT201909.create_resource({'$recursiveAnchor': False})
    resolver = Registry().with_resources([('http://example.com', three), ('http://example.com/foo/', two), ('http://example.com/foo/bar', one)]).resolver()
    first = resolver.lookup('http://example.com')
    second = first.resolver.lookup('foo/')
    resolver = second.resolver.lookup('bar').resolver
    fourth = referencing.jsonschema.lookup_recursive_ref(resolver=resolver)
    assert fourth.contents == two.contents