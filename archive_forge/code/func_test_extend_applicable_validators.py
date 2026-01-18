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
def test_extend_applicable_validators(self):
    """
        Extending a validator preserves its notion of applicable validators.
        """
    schema = {'$defs': {'test': {'type': 'number'}}, '$ref': '#/$defs/test', 'maximum': 1}
    draft4 = validators.Draft4Validator(schema)
    self.assertTrue(draft4.is_valid(37))
    Derived = validators.extend(validators.Draft4Validator)
    self.assertTrue(Derived(schema).is_valid(37))