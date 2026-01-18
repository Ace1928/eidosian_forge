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
def test_additionalItems(self):
    instance = ['foo', 1]
    schema = {'items': [], 'additionalItems': {'type': 'integer', 'minimum': 5}}
    validator = validators.Draft3Validator(schema)
    errors = validator.iter_errors(instance)
    e1, e2 = sorted_errors(errors)
    self.assertEqual(e1.path, deque([0]))
    self.assertEqual(e2.path, deque([1]))
    self.assertEqual(e1.json_path, '$[0]')
    self.assertEqual(e2.json_path, '$[1]')
    self.assertEqual(e1.validator, 'type')
    self.assertEqual(e2.validator, 'minimum')