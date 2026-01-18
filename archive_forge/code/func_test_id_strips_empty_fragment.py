from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_id_strips_empty_fragment(self):
    uri = 'http://example.com/'
    root = ID_AND_CHILDREN.create_resource({'ID': uri + '#'})
    assert root.id() == uri