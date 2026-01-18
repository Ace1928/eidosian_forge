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
def test_it_uses_best_match(self):
    schema = {'oneOf': [{'type': 'number', 'minimum': 20}, {'type': 'array'}]}
    with self.assertRaises(exceptions.ValidationError) as e:
        validators.validate(12, schema)
    self.assertIn('12 is less than the minimum of 20', str(e.exception))