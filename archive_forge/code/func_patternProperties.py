import re
from jsonschema import _utils
from jsonschema.exceptions import FormatError, ValidationError
from jsonschema.compat import iteritems
def patternProperties(validator, patternProperties, instance, schema):
    if not validator.is_type(instance, 'object'):
        return
    for pattern, subschema in iteritems(patternProperties):
        for k, v in iteritems(instance):
            if re.search(pattern, k):
                for error in validator.descend(v, subschema, path=k, schema_path=pattern):
                    yield error