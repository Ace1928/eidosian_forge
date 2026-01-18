from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_crawl_finds_anchors_no_id(self):
    resource = ID_AND_CHILDREN.create_resource({'anchors': {'foo': 12}})
    registry = Registry().with_resource('urn:root', resource)
    assert registry.crawl().anchor('urn:root', 'foo').value == Anchor(name='foo', resource=ID_AND_CHILDREN.create_resource(12))