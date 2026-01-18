from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_combine_with_common_retrieve(self):
    one = Resource.opaque(contents={})
    two = ID_AND_CHILDREN.create_resource({'foo': 'bar'})
    three = ID_AND_CHILDREN.create_resource({'baz': 'quux'})

    def retrieve(uri):
        pass
    first = Registry(retrieve=retrieve).with_resource('http://example.com/1', one)
    second = Registry(retrieve=retrieve).with_resource('http://example.com/2', two)
    third = Registry(retrieve=retrieve).with_resource('http://example.com/3', three)
    assert first.combine(second, third) == Registry(retrieve=retrieve).with_resources([('http://example.com/1', one), ('http://example.com/2', two), ('http://example.com/3', three)])
    assert second.combine(first, third) == Registry(retrieve=retrieve).with_resources([('http://example.com/1', one), ('http://example.com/2', two), ('http://example.com/3', three)])