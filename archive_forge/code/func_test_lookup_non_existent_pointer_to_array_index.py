from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_lookup_non_existent_pointer_to_array_index(self):
    resource = Resource.opaque([1, 2, 4, 8])
    resolver = Registry({'http://example.com/1': resource}).resolver()
    ref = 'http://example.com/1#/10'
    with pytest.raises(exceptions.Unresolvable) as e:
        resolver.lookup(ref)
    assert e.value == exceptions.PointerToNowhere(ref='/10', resource=resource)