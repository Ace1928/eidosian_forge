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
def test_import_FormatError(self):
    """
        As of v4.18.0, importing FormatError from the package root is
        deprecated in favor of doing so from jsonschema.exceptions.
        """
    message = 'Importing FormatError directly from the jsonschema package '
    with self.assertWarnsRegex(DeprecationWarning, message) as w:
        from jsonschema import FormatError
    self.assertEqual(FormatError, exceptions.FormatError)
    self.assertEqual(w.filename, __file__)