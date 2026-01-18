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
def test_helpful_error_message_on_failed_pop_scope(self):
    resolver = validators._RefResolver('', {})
    resolver.pop_scope()
    with self.assertRaises(exceptions._RefResolutionError) as exc:
        resolver.pop_scope()
    self.assertIn('Failed to pop the scope', str(exc.exception))