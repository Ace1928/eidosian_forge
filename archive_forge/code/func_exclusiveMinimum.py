from fractions import Fraction
import re
from jsonschema._utils import (
from jsonschema.exceptions import FormatError, ValidationError
def exclusiveMinimum(validator, minimum, instance, schema):
    if not validator.is_type(instance, 'number'):
        return
    if instance <= minimum:
        yield ValidationError(f'{instance!r} is less than or equal to the minimum of {minimum!r}')