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
def test_False_is_not_a_schema_even_if_you_forget_to_check(self):
    with self.assertRaises(Exception) as e:
        self.Validator(False).validate(12)
    self.assertNotIsInstance(e.exception, exceptions.ValidationError)