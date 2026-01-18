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
def test_RefResolver_in_scope(self):
    """
        As of v4.0.0, RefResolver.in_scope is deprecated.
        """
    resolver = validators._RefResolver.from_schema({})
    message = 'jsonschema.RefResolver.in_scope is deprecated '
    with self.assertWarnsRegex(DeprecationWarning, message) as w:
        with resolver.in_scope('foo'):
            pass
    self.assertEqual(w.filename, __file__)