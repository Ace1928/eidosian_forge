from collections import namedtuple
from unittest import TestCase
from jsonschema import ValidationError, _keywords
from jsonschema._types import TypeChecker
from jsonschema.exceptions import UndefinedTypeCheck, UnknownType
from jsonschema.validators import Draft202012Validator, extend
def is_object_or_named_tuple(checker, instance):
    if Draft202012Validator.TYPE_CHECKER.is_type(instance, 'object'):
        return True
    return is_namedtuple(instance)