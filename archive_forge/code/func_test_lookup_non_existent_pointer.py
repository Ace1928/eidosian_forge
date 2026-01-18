from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_lookup_non_existent_pointer(self):
    resource = Resource.opaque({'foo': {}})
    resolver = Registry({'http://example.com/1': resource}).resolver()
    ref = 'http://example.com/1#/foo/bar'
    with pytest.raises(exceptions.Unresolvable) as e:
        resolver.lookup(ref)
    assert e.value == exceptions.PointerToNowhere(ref='/foo/bar', resource=resource)
    assert str(e.value) == "'/foo/bar' does not exist within {'foo': {}}"