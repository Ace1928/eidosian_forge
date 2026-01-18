from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_no_such_resource(self):
    registry = Registry()
    with pytest.raises(exceptions.NoSuchResource) as e:
        registry['urn:bigboom']
    assert e.value == exceptions.NoSuchResource(ref='urn:bigboom')