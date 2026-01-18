from unittest import TestCase
from jsonschema import FormatChecker, ValidationError
from jsonschema.exceptions import FormatError
from jsonschema.validators import Draft4Validator
def test_format_checkers_come_with_defaults(self):
    checker = FormatChecker()
    with self.assertRaises(FormatError):
        checker.check(instance='not-an-ipv4', format='ipv4')