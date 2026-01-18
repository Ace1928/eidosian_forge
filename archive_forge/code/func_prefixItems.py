from fractions import Fraction
import re
from jsonschema._utils import (
from jsonschema.exceptions import FormatError, ValidationError
def prefixItems(validator, prefixItems, instance, schema):
    if not validator.is_type(instance, 'array'):
        return
    for (index, item), subschema in zip(enumerate(instance), prefixItems):
        yield from validator.descend(instance=item, schema=subschema, schema_path=index, path=index)