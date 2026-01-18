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
def test_import_ErrorTree(self):
    """
        As of v4.18.0, importing ErrorTree from the package root is
        deprecated in favor of doing so from jsonschema.exceptions.
        """
    message = 'Importing ErrorTree directly from the jsonschema package '
    with self.assertWarnsRegex(DeprecationWarning, message) as w:
        from jsonschema import ErrorTree
    self.assertEqual(ErrorTree, exceptions.ErrorTree)
    self.assertEqual(w.filename, __file__)