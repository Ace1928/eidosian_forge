from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_anchors_delegates_to_specification(self):
    resource = ID_AND_CHILDREN.create_resource({'anchors': {'foo': {}, 'bar': 1, 'baz': ''}})
    assert list(resource.anchors()) == [Anchor(name='foo', resource=ID_AND_CHILDREN.create_resource({})), Anchor(name='bar', resource=ID_AND_CHILDREN.create_resource(1)), Anchor(name='baz', resource=ID_AND_CHILDREN.create_resource(''))]