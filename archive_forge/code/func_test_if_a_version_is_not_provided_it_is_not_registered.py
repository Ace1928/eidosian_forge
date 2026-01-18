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
def test_if_a_version_is_not_provided_it_is_not_registered(self):
    original = dict(validators._META_SCHEMAS)
    validators.create(meta_schema={'id': 'id'})
    self.assertEqual(validators._META_SCHEMAS, original)