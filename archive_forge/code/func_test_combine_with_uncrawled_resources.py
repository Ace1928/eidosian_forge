from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_combine_with_uncrawled_resources(self):
    one = Resource.opaque(contents={})
    two = ID_AND_CHILDREN.create_resource({'foo': 'bar'})
    three = ID_AND_CHILDREN.create_resource({'baz': 'quux'})
    first = Registry().with_resource('http://example.com/1', one)
    second = Registry().with_resource('http://example.com/foo/bar', two)
    third = Registry({'http://example.com/1': one, 'http://example.com/baz': three})
    expected = Registry([('http://example.com/1', one), ('http://example.com/foo/bar', two), ('http://example.com/baz', three)])
    combined = first.combine(second, third)
    assert combined != expected
    assert combined.crawl() == expected