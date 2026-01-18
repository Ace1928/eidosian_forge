from fractions import Fraction
import re
from jsonschema._utils import (
from jsonschema.exceptions import FormatError, ValidationError
def dynamicRef(validator, dynamicRef, instance, schema):
    yield from validator._validate_reference(ref=dynamicRef, instance=instance)