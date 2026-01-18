from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_detect_unneeded_default(self):
    schema = {'$schema': 'https://json-schema.org/draft/2020-12/schema'}
    specification = Specification.OPAQUE.detect(schema)
    assert specification == DRAFT202012