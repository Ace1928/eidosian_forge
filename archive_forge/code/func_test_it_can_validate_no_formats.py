from unittest import TestCase
from jsonschema import FormatChecker, ValidationError
from jsonschema.exceptions import FormatError
from jsonschema.validators import Draft4Validator
def test_it_can_validate_no_formats(self):
    checker = FormatChecker(formats=())
    self.assertFalse(checker.checkers)