from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_matmul_many_resources(self):
    one_uri = 'urn:example:one'
    one = ID_AND_CHILDREN.create_resource({'ID': one_uri, 'foo': 12})
    two_uri = 'urn:example:two'
    two = ID_AND_CHILDREN.create_resource({'ID': two_uri, 'foo': 12})
    registry = [one, two] @ Registry()
    assert registry == Registry().with_resources([(one_uri, one), (two_uri, two)])