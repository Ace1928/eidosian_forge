from unittest import TestCase
from jsonschema import FormatChecker, ValidationError
from jsonschema.exceptions import FormatError
from jsonschema.validators import Draft4Validator
def test_format_error_causes_become_validation_error_causes(self):
    checker = FormatChecker()
    checker.checks('boom', raises=ValueError)(boom)
    validator = Draft4Validator({'format': 'boom'}, format_checker=checker)
    with self.assertRaises(ValidationError) as cm:
        validator.validate('BOOM')
    self.assertIs(cm.exception.cause, BOOM)
    self.assertIs(cm.exception.__cause__, BOOM)