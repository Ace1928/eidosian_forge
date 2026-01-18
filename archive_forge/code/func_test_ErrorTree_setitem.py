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
def test_ErrorTree_setitem(self):
    """
        As of v4.20.0, setting items on an ErrorTree is deprecated.
        """
    e = exceptions.ValidationError('some error', path=['foo'])
    tree = exceptions.ErrorTree()
    subtree = exceptions.ErrorTree(errors=[e])
    message = 'ErrorTree.__setitem__ is '
    with self.assertWarnsRegex(DeprecationWarning, message) as w:
        tree['foo'] = subtree
    self.assertEqual(tree['foo'], subtree)
    self.assertEqual(w.filename, __file__)