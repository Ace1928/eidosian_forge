from referencing import Registry
from referencing.jsonschema import DRAFT202012
from rpds import HashTrieMap, HashTrieSet
from jsonschema import Draft202012Validator
def registry_add():
    resource = DRAFT202012.create_resource(schema)
    return registry.with_resource(uri='urn:example', resource=resource)