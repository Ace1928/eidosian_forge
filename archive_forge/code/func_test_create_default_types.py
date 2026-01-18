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
def test_create_default_types(self):
    Validator = validators.create(meta_schema={}, validators=())
    self.assertTrue(all((Validator({}).is_type(instance=instance, type=type) for type, instance in [('array', []), ('boolean', True), ('integer', 12), ('null', None), ('number', 12.0), ('object', {}), ('string', 'foo')])))