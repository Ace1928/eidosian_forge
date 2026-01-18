from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_from_contents_with_no_discernible_information_and_default(self):
    resource = Resource.from_contents({'foo': 'bar'}, default_specification=Specification.OPAQUE)
    assert resource == Resource.opaque(contents={'foo': 'bar'})