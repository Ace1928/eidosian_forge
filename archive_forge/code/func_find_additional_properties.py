import itertools
import json
import pkgutil
import re
from jsonschema.compat import str_types, MutableMapping, urlsplit
def find_additional_properties(instance, schema):
    """
    Return the set of additional properties for the given ``instance``.

    Weeds out properties that should have been validated by ``properties`` and
    / or ``patternProperties``.

    Assumes ``instance`` is dict-like already.

    """
    properties = schema.get('properties', {})
    patterns = '|'.join(schema.get('patternProperties', {}))
    for property in instance:
        if property not in properties:
            if patterns and re.search(patterns, property):
                continue
            yield property