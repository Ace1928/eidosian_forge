import re
from referencing.jsonschema import lookup_recursive_ref
from jsonschema import _utils
from jsonschema.exceptions import ValidationError
def ignore_ref_siblings(schema):
    """
    Ignore siblings of ``$ref`` if it is present.

    Otherwise, return all keywords.

    Suitable for use with `create`'s ``applicable_validators`` argument.
    """
    ref = schema.get('$ref')
    if ref is not None:
        return [('$ref', ref)]
    else:
        return schema.items()