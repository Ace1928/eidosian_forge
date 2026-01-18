from contextlib import contextmanager
from io import BytesIO
from unittest import TestCase, mock
import importlib.metadata
import json
import subprocess
import sys
import urllib.request
import referencing.exceptions
from jsonschema import FormatChecker, exceptions, protocols, validators
def test_draftN_format_checker(self):
    """
        As of v4.16.0, accessing jsonschema.draftn_format_checker is deprecated
        in favor of Validator.FORMAT_CHECKER.
        """
    message = 'Accessing jsonschema.draft202012_format_checker is '
    with self.assertWarnsRegex(DeprecationWarning, message) as w:
        from jsonschema import draft202012_format_checker
    self.assertIs(draft202012_format_checker, validators.Draft202012Validator.FORMAT_CHECKER)
    self.assertEqual(w.filename, __file__)
    message = 'Accessing jsonschema.draft201909_format_checker is '
    with self.assertWarnsRegex(DeprecationWarning, message) as w:
        from jsonschema import draft201909_format_checker
    self.assertIs(draft201909_format_checker, validators.Draft201909Validator.FORMAT_CHECKER)
    self.assertEqual(w.filename, __file__)
    message = 'Accessing jsonschema.draft7_format_checker is '
    with self.assertWarnsRegex(DeprecationWarning, message) as w:
        from jsonschema import draft7_format_checker
    self.assertIs(draft7_format_checker, validators.Draft7Validator.FORMAT_CHECKER)
    self.assertEqual(w.filename, __file__)
    message = 'Accessing jsonschema.draft6_format_checker is '
    with self.assertWarnsRegex(DeprecationWarning, message) as w:
        from jsonschema import draft6_format_checker
    self.assertIs(draft6_format_checker, validators.Draft6Validator.FORMAT_CHECKER)
    self.assertEqual(w.filename, __file__)
    message = 'Accessing jsonschema.draft4_format_checker is '
    with self.assertWarnsRegex(DeprecationWarning, message) as w:
        from jsonschema import draft4_format_checker
    self.assertIs(draft4_format_checker, validators.Draft4Validator.FORMAT_CHECKER)
    self.assertEqual(w.filename, __file__)
    message = 'Accessing jsonschema.draft3_format_checker is '
    with self.assertWarnsRegex(DeprecationWarning, message) as w:
        from jsonschema import draft3_format_checker
    self.assertIs(draft3_format_checker, validators.Draft3Validator.FORMAT_CHECKER)
    self.assertEqual(w.filename, __file__)
    with self.assertRaises(ImportError):
        from jsonschema import draft1234_format_checker