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
def test_FormatChecker_cls_checks(self):
    """
        As of v4.14.0, FormatChecker.cls_checks is deprecated without
        replacement.
        """
    self.addCleanup(FormatChecker.checkers.pop, 'boom', None)
    message = 'FormatChecker.cls_checks '
    with self.assertWarnsRegex(DeprecationWarning, message) as w:
        FormatChecker.cls_checks('boom')
    self.assertEqual(w.filename, __file__)