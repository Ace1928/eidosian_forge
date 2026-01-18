from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
@pytest.mark.parametrize('thing', THINGS)
def test_no_subresources(self, thing):
    """
        An arbitrary thing has no subresources.
        """
    assert list(Specification.OPAQUE.subresources_of(thing)) == []