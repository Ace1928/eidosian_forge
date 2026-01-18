import jsonschema
from jsonschema import exceptions as schema_exc
def schema_validate(data, schema):
    """Validates given data using provided json schema."""
    Validator = jsonschema.validators.validator_for(schema)
    type_checker = Validator.TYPE_CHECKER.redefine('array', lambda checker, data: isinstance(data, (list, tuple)))
    TupleAllowingValidator = jsonschema.validators.extend(Validator, type_checker=type_checker)
    TupleAllowingValidator(schema).validate(data)