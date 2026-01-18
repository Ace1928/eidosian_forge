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
def test_check_schema_with_different_metaschema(self):
    """
        One can create a validator class whose metaschema uses a different
        dialect than itself.
        """
    NoEmptySchemasValidator = validators.create(meta_schema={'$schema': validators.Draft202012Validator.META_SCHEMA['$id'], 'not': {'const': {}}})
    NoEmptySchemasValidator.check_schema({'foo': 'bar'})
    with self.assertRaises(exceptions.SchemaError):
        NoEmptySchemasValidator.check_schema({})
    NoEmptySchemasValidator({'foo': 'bar'}).validate('foo')