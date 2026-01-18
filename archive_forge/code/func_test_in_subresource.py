from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_in_subresource(self):
    root = ID_AND_CHILDREN.create_resource({'ID': 'http://example.com/', 'children': [{'ID': 'child/', 'children': [{'ID': 'grandchild'}]}]})
    registry = root @ Registry()
    resolver = registry.resolver()
    first = resolver.lookup('http://example.com/')
    assert first.contents == root.contents
    with pytest.raises(exceptions.Unresolvable):
        first.resolver.lookup('grandchild')
    sub = first.resolver.in_subresource(ID_AND_CHILDREN.create_resource(first.contents['children'][0]))
    second = sub.lookup('grandchild')
    assert second.contents == {'ID': 'grandchild'}