from collections import namedtuple
from unittest import TestCase
from jsonschema import ValidationError, _keywords
from jsonschema._types import TypeChecker
from jsonschema.exceptions import UndefinedTypeCheck, UnknownType
from jsonschema.validators import Draft202012Validator, extend
def test_checks_can_be_added_at_init(self):
    checker = TypeChecker({'two': equals_2})
    self.assertEqual(checker, TypeChecker().redefine('two', equals_2))