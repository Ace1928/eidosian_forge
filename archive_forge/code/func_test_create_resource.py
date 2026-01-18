from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_create_resource(self):
    specification = Specification(name='', id_of=lambda contents: 'urn:fixedID', subresources_of=lambda contents: [], anchors_in=lambda specification, contents: [], maybe_in_subresource=lambda segments, resolver, subresource: resolver)
    resource = specification.create_resource(contents={'foo': 'baz'})
    assert resource == Resource(contents={'foo': 'baz'}, specification=specification)
    assert resource.id() == 'urn:fixedID'