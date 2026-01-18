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
def test_Validator_is_valid_two_arguments(self):
    """
        As of v4.0.0, calling is_valid with two arguments (to provide a
        different schema) is deprecated.
        """
    validator = validators.Draft7Validator({})
    message = 'Passing a schema to Validator.is_valid is deprecated '
    with self.assertWarnsRegex(DeprecationWarning, message) as w:
        result = validator.is_valid('foo', {'type': 'number'})
    self.assertFalse(result)
    self.assertEqual(w.filename, __file__)