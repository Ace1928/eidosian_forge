from fractions import Fraction
import re
from jsonschema._utils import (
from jsonschema.exceptions import FormatError, ValidationError
def exclusiveMaximum(validator, maximum, instance, schema):
    if not validator.is_type(instance, 'number'):
        return
    if instance >= maximum:
        yield ValidationError(f'{instance!r} is greater than or equal to the maximum of {maximum!r}')