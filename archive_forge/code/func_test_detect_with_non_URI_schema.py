from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_detect_with_non_URI_schema(self):
    with pytest.raises(exceptions.CannotDetermineSpecification):
        Specification.detect({'$schema': 37})