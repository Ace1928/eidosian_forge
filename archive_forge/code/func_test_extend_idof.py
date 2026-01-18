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
def test_extend_idof(self):
    """
        Extending a validator preserves its notion of schema IDs.
        """

    def id_of(schema):
        return schema.get('__test__', self.Validator.ID_OF(schema))
    correct_id = 'the://correct/id/'
    meta_schema = {'$id': 'the://wrong/id/', '__test__': correct_id}
    Original = validators.create(meta_schema=meta_schema, validators=self.validators, type_checker=self.type_checker, id_of=id_of)
    self.assertEqual(Original.ID_OF(Original.META_SCHEMA), correct_id)
    Derived = validators.extend(Original)
    self.assertEqual(Derived.ID_OF(Derived.META_SCHEMA), correct_id)