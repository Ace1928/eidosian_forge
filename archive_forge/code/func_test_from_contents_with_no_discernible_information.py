from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_from_contents_with_no_discernible_information(self):
    """
        Creating a resource with no discernible way to see what
        specification it belongs to (e.g. no ``$schema`` keyword for JSON
        Schema) raises an error.
        """
    with pytest.raises(exceptions.CannotDetermineSpecification):
        Resource.from_contents({'foo': 'bar'})