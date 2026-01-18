from fractions import Fraction
import re
from jsonschema._utils import (
from jsonschema.exceptions import FormatError, ValidationError
def dependentRequired(validator, dependentRequired, instance, schema):
    if not validator.is_type(instance, 'object'):
        return
    for property, dependency in dependentRequired.items():
        if property not in instance:
            continue
        for each in dependency:
            if each not in instance:
                message = f'{each!r} is a dependency of {property!r}'
                yield ValidationError(message)