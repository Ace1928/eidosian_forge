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
def test_newly_created_validator_with_ref_resolver(self):
    """
        See https://github.com/python-jsonschema/jsonschema/issues/1061#issuecomment-1624266555.
        """

    def handle(uri):
        self.assertEqual(uri, 'http://example.com/foo')
        return {'type': 'integer'}
    resolver = validators._RefResolver('', {}, handlers={'http': handle})
    Validator = validators.create(meta_schema={}, validators=validators.Draft4Validator.VALIDATORS)
    schema = {'$id': 'http://example.com/bar', '$ref': 'foo'}
    validator = Validator(schema, resolver=resolver)
    self.assertEqual((validator.is_valid({}), validator.is_valid(37)), (False, True))