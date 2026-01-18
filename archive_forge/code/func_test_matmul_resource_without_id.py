from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_matmul_resource_without_id(self):
    resource = Resource.opaque(contents={'foo': 'bar'})
    with pytest.raises(exceptions.NoInternalID) as e:
        resource @ Registry()
    assert e.value == exceptions.NoInternalID(resource=resource)