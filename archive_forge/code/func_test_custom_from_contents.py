from functools import lru_cache
import json
import pytest
from referencing import Registry, Resource, exceptions
from referencing.jsonschema import DRAFT202012
from referencing.retrieval import to_cached_resource
def test_custom_from_contents(self):
    contents = {}
    stack = [json.dumps(contents)]

    @to_cached_resource(from_contents=DRAFT202012.create_resource)
    def retrieve(uri):
        return stack.pop()
    registry = Registry(retrieve=retrieve)
    expected = DRAFT202012.create_resource(contents)
    got = registry.get_or_retrieve('urn:example:schema')
    assert got.value == expected
    again = registry.get_or_retrieve('urn:example:schema')
    assert again.value is got.value