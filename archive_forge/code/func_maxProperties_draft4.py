import re
from jsonschema import _utils
from jsonschema.exceptions import FormatError, ValidationError
from jsonschema.compat import iteritems
def maxProperties_draft4(validator, mP, instance, schema):
    if not validator.is_type(instance, 'object'):
        return
    if validator.is_type(instance, 'object') and len(instance) > mP:
        yield ValidationError('%r has too many properties' % (instance,))