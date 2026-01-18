import re
from jsonschema import _utils
from jsonschema.exceptions import FormatError, ValidationError
from jsonschema.compat import iteritems
def properties_draft3(validator, properties, instance, schema):
    if not validator.is_type(instance, 'object'):
        return
    for property, subschema in iteritems(properties):
        if property in instance:
            for error in validator.descend(instance[property], subschema, path=property, schema_path=property):
                yield error
        elif subschema.get('required', False):
            error = ValidationError('%r is a required property' % property)
            error._set(validator='required', validator_value=subschema['required'], instance=instance, schema=schema)
            error.path.appendleft(property)
            error.schema_path.extend([property, 'required'])
            yield error