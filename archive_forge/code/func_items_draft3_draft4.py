import re
from referencing.jsonschema import lookup_recursive_ref
from jsonschema import _utils
from jsonschema.exceptions import ValidationError
def items_draft3_draft4(validator, items, instance, schema):
    if not validator.is_type(instance, 'array'):
        return
    if validator.is_type(items, 'object'):
        for index, item in enumerate(instance):
            yield from validator.descend(item, items, path=index)
    else:
        for (index, item), subschema in zip(enumerate(instance), items):
            yield from validator.descend(item, subschema, path=index, schema_path=index)