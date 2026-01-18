import re
from referencing.jsonschema import lookup_recursive_ref
from jsonschema import _utils
from jsonschema.exceptions import ValidationError
def dependencies_draft4_draft6_draft7(validator, dependencies, instance, schema):
    """
    Support for the ``dependencies`` keyword from pre-draft 2019-09.

    In later drafts, the keyword was split into separate
    ``dependentRequired`` and ``dependentSchemas`` validators.
    """
    if not validator.is_type(instance, 'object'):
        return
    for property, dependency in dependencies.items():
        if property not in instance:
            continue
        if validator.is_type(dependency, 'array'):
            for each in dependency:
                if each not in instance:
                    message = f'{each!r} is a dependency of {property!r}'
                    yield ValidationError(message)
        else:
            yield from validator.descend(instance, dependency, schema_path=property)