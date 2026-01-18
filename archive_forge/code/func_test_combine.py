from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_combine(self):
    one = Resource.opaque(contents={})
    two = ID_AND_CHILDREN.create_resource({'foo': 'bar'})
    three = ID_AND_CHILDREN.create_resource({'baz': 'quux'})
    four = ID_AND_CHILDREN.create_resource({'anchors': {'foo': 12}})
    first = Registry({'http://example.com/1': one})
    second = Registry().with_resource('http://example.com/foo/bar', two)
    third = Registry({'http://example.com/1': one, 'http://example.com/baz': three})
    fourth = Registry().with_resource('http://example.com/foo/quux', four).crawl()
    assert first.combine(second, third, fourth) == Registry([('http://example.com/1', one), ('http://example.com/baz', three), ('http://example.com/foo/quux', four)], anchors=HashTrieMap({('http://example.com/foo/quux', 'foo'): Anchor(name='foo', resource=ID_AND_CHILDREN.create_resource(12))})).with_resource('http://example.com/foo/bar', two)