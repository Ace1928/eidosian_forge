from collections import namedtuple
from unittest import TestCase
from jsonschema import ValidationError, _keywords
from jsonschema._types import TypeChecker
from jsonschema.exceptions import UndefinedTypeCheck, UnknownType
from jsonschema.validators import Draft202012Validator, extend
def test_is_unknown_type(self):
    with self.assertRaises(UndefinedTypeCheck) as e:
        TypeChecker().is_type(4, 'foobar')
    self.assertIn("'foobar' is unknown to this type checker", str(e.exception))
    self.assertTrue(e.exception.__suppress_context__, msg='Expected the internal KeyError to be hidden.')