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
def test_anyOf(self):
    instance = 5
    schema = {'anyOf': [{'minimum': 20}, {'type': 'string'}]}
    validator = validators.Draft4Validator(schema)
    errors = list(validator.iter_errors(instance))
    self.assertEqual(len(errors), 1)
    e = errors[0]
    self.assertEqual(e.validator, 'anyOf')
    self.assertEqual(e.validator_value, schema['anyOf'])
    self.assertEqual(e.instance, instance)
    self.assertEqual(e.schema, schema)
    self.assertIsNone(e.parent)
    self.assertEqual(e.path, deque([]))
    self.assertEqual(e.relative_path, deque([]))
    self.assertEqual(e.absolute_path, deque([]))
    self.assertEqual(e.json_path, '$')
    self.assertEqual(e.schema_path, deque(['anyOf']))
    self.assertEqual(e.relative_schema_path, deque(['anyOf']))
    self.assertEqual(e.absolute_schema_path, deque(['anyOf']))
    self.assertEqual(len(e.context), 2)
    e1, e2 = sorted_errors(e.context)
    self.assertEqual(e1.validator, 'minimum')
    self.assertEqual(e1.validator_value, schema['anyOf'][0]['minimum'])
    self.assertEqual(e1.instance, instance)
    self.assertEqual(e1.schema, schema['anyOf'][0])
    self.assertIs(e1.parent, e)
    self.assertEqual(e1.path, deque([]))
    self.assertEqual(e1.absolute_path, deque([]))
    self.assertEqual(e1.relative_path, deque([]))
    self.assertEqual(e1.json_path, '$')
    self.assertEqual(e1.schema_path, deque([0, 'minimum']))
    self.assertEqual(e1.relative_schema_path, deque([0, 'minimum']))
    self.assertEqual(e1.absolute_schema_path, deque(['anyOf', 0, 'minimum']))
    self.assertFalse(e1.context)
    self.assertEqual(e2.validator, 'type')
    self.assertEqual(e2.validator_value, schema['anyOf'][1]['type'])
    self.assertEqual(e2.instance, instance)
    self.assertEqual(e2.schema, schema['anyOf'][1])
    self.assertIs(e2.parent, e)
    self.assertEqual(e2.path, deque([]))
    self.assertEqual(e2.relative_path, deque([]))
    self.assertEqual(e2.absolute_path, deque([]))
    self.assertEqual(e2.json_path, '$')
    self.assertEqual(e2.schema_path, deque([1, 'type']))
    self.assertEqual(e2.relative_schema_path, deque([1, 'type']))
    self.assertEqual(e2.absolute_schema_path, deque(['anyOf', 1, 'type']))
    self.assertEqual(len(e2.context), 0)