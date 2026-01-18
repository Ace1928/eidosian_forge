from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_lookup_non_existent_anchor(self):
    root = ID_AND_CHILDREN.create_resource({'anchors': {}})
    resolver = Registry().with_resource('urn:example', root).resolver()
    resolved = resolver.lookup('urn:example')
    assert resolved.contents == root.contents
    ref = 'urn:example#noSuchAnchor'
    with pytest.raises(exceptions.Unresolvable) as e:
        resolver.lookup(ref)
    assert "'noSuchAnchor' does not exist" in str(e.value)
    assert e.value == exceptions.NoSuchAnchor(ref='urn:example', resource=root, anchor='noSuchAnchor')