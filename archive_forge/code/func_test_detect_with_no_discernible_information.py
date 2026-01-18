from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_detect_with_no_discernible_information(self):
    with pytest.raises(exceptions.CannotDetermineSpecification):
        Specification.detect({'foo': 'bar'})