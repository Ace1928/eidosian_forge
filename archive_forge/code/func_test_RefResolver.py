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
def test_RefResolver(self):
    """
        As of v4.18.0, RefResolver is fully deprecated.
        """
    message = 'jsonschema.RefResolver is deprecated'
    with self.assertWarnsRegex(DeprecationWarning, message) as w:
        from jsonschema import RefResolver
    self.assertEqual(w.filename, __file__)
    with self.assertWarnsRegex(DeprecationWarning, message) as w:
        from jsonschema.validators import RefResolver
    self.assertEqual(w.filename, __file__)