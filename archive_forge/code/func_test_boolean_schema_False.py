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
def test_boolean_schema_False(self):
    validator = validators.Draft7Validator(False)
    error, = validator.iter_errors(12)
    self.assertEqual((error.message, error.validator, error.validator_value, error.instance, error.schema, error.schema_path, error.json_path), ('False schema does not allow 12', None, None, 12, False, deque([]), '$'))