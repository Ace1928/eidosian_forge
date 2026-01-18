from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
@pytest.mark.parametrize('thing', THINGS)
def test_no_anchors(self, thing):
    """
        An arbitrary thing has no anchors.
        """
    assert list(Specification.OPAQUE.anchors_in(thing)) == []