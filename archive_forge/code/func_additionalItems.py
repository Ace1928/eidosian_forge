import re
from jsonschema import _utils
from jsonschema.exceptions import FormatError, ValidationError
from jsonschema.compat import iteritems
def additionalItems(validator, aI, instance, schema):
    if not validator.is_type(instance, 'array') or validator.is_type(schema.get('items', {}), 'object'):
        return
    len_items = len(schema.get('items', []))
    if validator.is_type(aI, 'object'):
        for index, item in enumerate(instance[len_items:], start=len_items):
            for error in validator.descend(item, aI, path=index):
                yield error
    elif not aI and len(instance) > len(schema.get('items', [])):
        error = 'Additional items are not allowed (%s %s unexpected)'
        yield ValidationError(error % _utils.extras_msg(instance[len(schema.get('items', [])):]))