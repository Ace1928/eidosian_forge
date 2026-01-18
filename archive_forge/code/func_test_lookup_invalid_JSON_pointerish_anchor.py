from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_lookup_invalid_JSON_pointerish_anchor(self):
    resolver = Registry().resolver_with_root(ID_AND_CHILDREN.create_resource({'ID': 'http://example.com/', 'foo': {'bar': 12}}))
    valid = resolver.lookup('#/foo/bar')
    assert valid.contents == 12
    with pytest.raises(exceptions.InvalidAnchor) as e:
        resolver.lookup('#foo/bar')
    assert " '#/foo/bar'" in str(e.value)