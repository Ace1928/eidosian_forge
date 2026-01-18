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
def test_WrappedReferencingError_hashability(self):
    """
        Ensure the wrapped referencing errors are hashable when possible.
        """
    with self.assertWarns(DeprecationWarning):
        from jsonschema import RefResolutionError
    validator = validators.Draft202012Validator({'$ref': 'urn:nothing'})
    with self.assertRaises(referencing.exceptions.Unresolvable) as u:
        validator.validate(12)
    with self.assertRaises(RefResolutionError) as e:
        validator.validate(12)
    self.assertIn(e.exception, {u.exception})
    self.assertIn(u.exception, {e.exception})