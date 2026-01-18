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
def test_contains_too_many(self):
    """
        `contains` + `maxContains` produces only one error, even if there are
        many more incorrectly matching elements.
        """
    schema = {'contains': {'type': 'string'}, 'maxContains': 2}
    validator = validators.Draft202012Validator(schema)
    error, = validator.iter_errors(['foo', 2, 'bar', 4, 'baz', 'quux'])
    self.assertEqual((error.message, error.validator, error.validator_value, error.instance, error.absolute_path, error.schema, error.schema_path, error.json_path), ('Too many items match the given schema (expected at most 2)', 'maxContains', 2, ['foo', 2, 'bar', 4, 'baz', 'quux'], deque([]), {'contains': {'type': 'string'}, 'maxContains': 2}, deque(['contains']), '$'))