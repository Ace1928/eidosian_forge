import pytest
from referencing import Registry, Resource, Specification
import referencing.jsonschema
def test_multiple_lookup_recursive_ref_to_bool():
    TRUE = referencing.jsonschema.DRAFT201909.create_resource(True)
    root = referencing.jsonschema.DRAFT201909.create_resource({'$id': 'http://example.com', '$recursiveAnchor': True, '$defs': {'foo': {'$id': 'foo', '$recursiveAnchor': True, '$defs': {'bar': True, 'baz': {'$recursiveAnchor': True, '$anchor': 'fooAnchor'}}}}})
    resolver = Registry().with_resources([('http://example.com', root), ('http://example.com/foo/', TRUE), ('http://example.com/foo/bar', root)]).resolver()
    first = resolver.lookup('http://example.com')
    second = first.resolver.lookup('foo/')
    resolver = second.resolver.lookup('bar').resolver
    fourth = referencing.jsonschema.lookup_recursive_ref(resolver=resolver)
    assert fourth.contents == root.contents