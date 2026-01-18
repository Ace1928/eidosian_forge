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
def test_contains_too_few_both_constrained(self):
    message = self.message_for(instance=['foo', 1], schema={'contains': {'type': 'string'}, 'minContains': 2, 'maxContains': 4})
    self.assertEqual(message, 'Too few items match the given schema (expected at least 2 but only 1 matched)')