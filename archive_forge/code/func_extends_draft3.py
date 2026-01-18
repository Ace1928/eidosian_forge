import re
from jsonschema import _utils
from jsonschema.exceptions import FormatError, ValidationError
from jsonschema.compat import iteritems
def extends_draft3(validator, extends, instance, schema):
    if validator.is_type(extends, 'object'):
        for error in validator.descend(instance, extends):
            yield error
        return
    for index, subschema in enumerate(extends):
        for error in validator.descend(instance, subschema, schema_path=index):
            yield error