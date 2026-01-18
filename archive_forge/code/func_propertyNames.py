from fractions import Fraction
import re
from jsonschema._utils import (
from jsonschema.exceptions import FormatError, ValidationError
def propertyNames(validator, propertyNames, instance, schema):
    if not validator.is_type(instance, 'object'):
        return
    for property in instance:
        yield from validator.descend(instance=property, schema=propertyNames)