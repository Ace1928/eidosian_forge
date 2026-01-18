from fractions import Fraction
import re
from jsonschema._utils import (
from jsonschema.exceptions import FormatError, ValidationError
def dependentSchemas(validator, dependentSchemas, instance, schema):
    if not validator.is_type(instance, 'object'):
        return
    for property, dependency in dependentSchemas.items():
        if property not in instance:
            continue
        yield from validator.descend(instance, dependency, schema_path=property)