from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_combine_self(self):
    """
        Combining a registry with itself short-circuits.

        This is a performance optimization -- otherwise we do lots more work
        (in jsonschema this seems to correspond to making the test suite take
         *3x* longer).
        """
    registry = Registry({'urn:foo': 'bar'})
    assert registry.combine(registry) is registry