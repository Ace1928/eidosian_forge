from itertools import cycle
from jsonschema.validators import validator_for
def nested_schema(levels):
    """
    Produce a schema which validates deeply nested objects and arrays.
    """
    names = cycle(['foo', 'bar', 'baz', 'quux', 'spam', 'eggs'])
    schema = {'type': 'object', 'properties': {'ham': {'type': 'string'}}}
    for _, name in zip(range(levels - 1), names):
        schema = {'type': 'object', 'properties': {name: schema}}
    return schema