from __future__ import annotations
from collections import deque, namedtuple
from contextlib import contextmanager
from decimal import Decimal
from io import BytesIO
from typing import Any
from unittest import TestCase, mock
from urllib.request import pathname2url
import json
import os
import sys
import tempfile
import warnings
from attrs import define, field
from referencing.jsonschema import DRAFT202012
import referencing.exceptions
from jsonschema import (
def test_it_properly_formats_tuples_in_errors(self):
    """
        A tuple instance properly formats validation errors for uniqueItems.

        See #224
        """
    TupleValidator = validators.extend(self.Validator, type_checker=self.Validator.TYPE_CHECKER.redefine('array', lambda checker, thing: isinstance(thing, tuple)))
    with self.assertRaises(exceptions.ValidationError) as e:
        TupleValidator({'uniqueItems': True}).validate((1, 1))
    self.assertIn('(1, 1) has non-unique elements', str(e.exception))