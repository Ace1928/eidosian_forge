import re
from jsonschema import _utils
from jsonschema.exceptions import FormatError, ValidationError
from jsonschema.compat import iteritems
def properties_draft4(validator, properties, instance, schema):
    if not validator.is_type(instance, 'object'):
        return
    for property, subschema in iteritems(properties):
        if property in instance:
            for error in validator.descend(instance[property], subschema, path=property, schema_path=property):
                yield error