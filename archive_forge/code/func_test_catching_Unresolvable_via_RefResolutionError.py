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
def test_catching_Unresolvable_via_RefResolutionError(self):
    """
        Until RefResolutionError is removed, it is still possible to catch
        exceptions from reference resolution using it, even though they may
        have been raised by referencing.
        """
    with self.assertWarns(DeprecationWarning):
        from jsonschema import RefResolutionError
    validator = validators.Draft202012Validator({'$ref': 'urn:nothing'})
    with self.assertRaises(referencing.exceptions.Unresolvable) as u:
        validator.validate(12)
    with self.assertRaises(RefResolutionError) as e:
        validator.validate(12)
    self.assertEqual((e.exception, str(e.exception)), (u.exception, 'Unresolvable: urn:nothing'))