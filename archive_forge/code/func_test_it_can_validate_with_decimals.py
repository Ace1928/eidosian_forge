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
def test_it_can_validate_with_decimals(self):
    schema = {'items': {'type': 'number'}}
    Validator = validators.extend(self.Validator, type_checker=self.Validator.TYPE_CHECKER.redefine('number', lambda checker, thing: isinstance(thing, (int, float, Decimal)) and (not isinstance(thing, bool))))
    validator = Validator(schema)
    validator.validate([1, 1.1, Decimal(1) / Decimal(8)])
    invalid = ['foo', {}, [], True, None]
    self.assertEqual([error.instance for error in validator.iter_errors(invalid)], invalid)