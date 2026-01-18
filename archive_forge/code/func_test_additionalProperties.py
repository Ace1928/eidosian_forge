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
def test_additionalProperties(self):
    instance = {'bar': 'bar', 'foo': 2}
    schema = {'additionalProperties': {'type': 'integer', 'minimum': 5}}
    validator = validators.Draft3Validator(schema)
    errors = validator.iter_errors(instance)
    e1, e2 = sorted_errors(errors)
    self.assertEqual(e1.path, deque(['bar']))
    self.assertEqual(e2.path, deque(['foo']))
    self.assertEqual(e1.json_path, '$.bar')
    self.assertEqual(e2.json_path, '$.foo')
    self.assertEqual(e1.validator, 'type')
    self.assertEqual(e2.validator, 'minimum')