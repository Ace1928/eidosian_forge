import re
from jsonschema import _utils
from jsonschema.exceptions import FormatError, ValidationError
from jsonschema.compat import iteritems
def disallow_draft3(validator, disallow, instance, schema):
    for disallowed in _utils.ensure_list(disallow):
        if validator.is_valid(instance, {'type': [disallowed]}):
            yield ValidationError('%r is disallowed for %r' % (disallowed, instance))