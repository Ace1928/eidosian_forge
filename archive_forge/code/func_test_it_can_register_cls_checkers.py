from unittest import TestCase
from jsonschema import FormatChecker, ValidationError
from jsonschema.exceptions import FormatError
from jsonschema.validators import Draft4Validator
def test_it_can_register_cls_checkers(self):
    original = dict(FormatChecker.checkers)
    self.addCleanup(FormatChecker.checkers.pop, 'boom')
    with self.assertWarns(DeprecationWarning):
        FormatChecker.cls_checks('boom')(boom)
    self.assertEqual(FormatChecker.checkers, dict(original, boom=(boom, ())))