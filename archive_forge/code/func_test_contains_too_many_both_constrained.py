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
def test_contains_too_many_both_constrained(self):
    message = self.message_for(instance=['foo'] * 5, schema={'contains': {'type': 'string'}, 'minContains': 2, 'maxContains': 4})
    self.assertEqual(message, 'Too many items match the given schema (expected at most 4)')