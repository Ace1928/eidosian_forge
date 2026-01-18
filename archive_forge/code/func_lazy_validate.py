from keystone.common.validation import validators
def lazy_validate(request_body_schema, resource_to_validate):
    """A non-decorator way to validate a request, to be used inline.

    :param request_body_schema: a schema to validate the resource reference
    :param resource_to_validate: dictionary to validate
    :raises keystone.exception.ValidationError: if `resource_to_validate` is
            None. (see wrapper method below).
    :raises TypeError: at decoration time when the expected resource to
                       validate isn't found in the decorated method's
                       signature

    """
    schema_validator = validators.SchemaValidator(request_body_schema)
    schema_validator.validate(resource_to_validate)