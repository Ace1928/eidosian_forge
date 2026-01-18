import re
from jsonschema import _utils
from jsonschema.exceptions import FormatError, ValidationError
from jsonschema.compat import iteritems
def allOf_draft4(validator, allOf, instance, schema):
    for index, subschema in enumerate(allOf):
        for error in validator.descend(instance, subschema, schema_path=index):
            yield error