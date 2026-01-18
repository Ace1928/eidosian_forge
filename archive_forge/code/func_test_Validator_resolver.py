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
def test_Validator_resolver(self):
    """
        As of v4.18.0, accessing Validator.resolver is deprecated.
        """
    validator = validators.Draft7Validator({})
    message = 'Accessing Draft7Validator.resolver is '
    with self.assertWarnsRegex(DeprecationWarning, message) as w:
        self.assertIsInstance(validator.resolver, validators._RefResolver)
    self.assertEqual(w.filename, __file__)