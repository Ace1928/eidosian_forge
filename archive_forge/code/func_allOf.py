from fractions import Fraction
import re
from jsonschema._utils import (
from jsonschema.exceptions import FormatError, ValidationError
def allOf(validator, allOf, instance, schema):
    for index, subschema in enumerate(allOf):
        yield from validator.descend(instance, subschema, schema_path=index)