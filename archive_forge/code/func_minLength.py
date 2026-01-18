import re
from jsonschema import _utils
from jsonschema.exceptions import FormatError, ValidationError
from jsonschema.compat import iteritems
def minLength(validator, mL, instance, schema):
    if validator.is_type(instance, 'string') and len(instance) < mL:
        yield ValidationError('%r is too short' % (instance,))