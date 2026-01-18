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
def test_check_redefined_sequence(self):
    """
        Allow array to validate against another defined sequence type
        """
    schema = {'type': 'array', 'uniqueItems': True}
    MyMapping = namedtuple('MyMapping', 'a, b')
    Validator = validators.extend(self.Validator, type_checker=self.Validator.TYPE_CHECKER.redefine_many({'array': lambda checker, thing: isinstance(thing, (list, deque)), 'object': lambda checker, thing: isinstance(thing, (dict, MyMapping))}))
    validator = Validator(schema)
    valid_instances = [deque(['a', None, '1', '', True]), deque([[False], [0]]), [deque([False]), deque([0])], [[deque([False])], [deque([0])]], [[[[[deque([False])]]]], [[[[deque([0])]]]]], [deque([deque([False])]), deque([deque([0])])], [MyMapping('a', 0), MyMapping('a', False)], [MyMapping('a', [deque([0])]), MyMapping('a', [deque([False])])], [MyMapping('a', [MyMapping('a', deque([0]))]), MyMapping('a', [MyMapping('a', deque([False]))])], [deque(deque(deque([False]))), deque(deque(deque([0])))]]
    for instance in valid_instances:
        validator.validate(instance)
    invalid_instances = [deque(['a', 'b', 'a']), deque([[False], [False]]), [deque([False]), deque([False])], [[deque([False])], [deque([False])]], [[[[[deque([False])]]]], [[[[deque([False])]]]]], [deque([deque([False])]), deque([deque([False])])], [MyMapping('a', False), MyMapping('a', False)], [MyMapping('a', [deque([False])]), MyMapping('a', [deque([False])])], [MyMapping('a', [MyMapping('a', deque([False]))]), MyMapping('a', [MyMapping('a', deque([False]))])], [deque(deque(deque([False]))), deque(deque(deque([False])))]]
    for instance in invalid_instances:
        with self.assertRaises(exceptions.ValidationError):
            validator.validate(instance)