from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_lookup_subresource(self):
    root = ID_AND_CHILDREN.create_resource({'ID': 'http://example.com/', 'children': [{'ID': 'http://example.com/a', 'foo': 12}]})
    registry = root @ Registry()
    resolved = registry.resolver().lookup('http://example.com/a')
    assert resolved.contents == {'ID': 'http://example.com/a', 'foo': 12}