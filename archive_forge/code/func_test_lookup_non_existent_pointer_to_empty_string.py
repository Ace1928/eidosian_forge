from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_lookup_non_existent_pointer_to_empty_string(self):
    resource = Resource.opaque({'foo': {}})
    resolver = Registry().resolver_with_root(resource)
    with pytest.raises(exceptions.Unresolvable, match="^'/' does not exist within {'foo': {}}.*'#'") as e:
        resolver.lookup('#/')
    assert e.value == exceptions.PointerToNowhere(ref='/', resource=resource)