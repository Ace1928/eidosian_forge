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
def test_does_not_warn_if_meta_schema_is_unspecified(self):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        validators.validator_for(schema={}, default={})
    self.assertFalse(w)