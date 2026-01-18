from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_lookup_retrieved_resource(self):
    resource = Resource.opaque(contents={'foo': 'baz'})
    resolver = Registry(retrieve=lambda uri: resource).resolver()
    resolved = resolver.lookup('http://example.com/')
    assert resolved.contents == resource.contents