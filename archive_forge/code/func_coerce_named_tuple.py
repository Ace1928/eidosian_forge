from collections import namedtuple
from unittest import TestCase
from jsonschema import ValidationError, _keywords
from jsonschema._types import TypeChecker
from jsonschema.exceptions import UndefinedTypeCheck, UnknownType
from jsonschema.validators import Draft202012Validator, extend
def coerce_named_tuple(fn):

    def coerced(validator, value, instance, schema):
        if is_namedtuple(instance):
            instance = instance._asdict()
        return fn(validator, value, instance, schema)
    return coerced