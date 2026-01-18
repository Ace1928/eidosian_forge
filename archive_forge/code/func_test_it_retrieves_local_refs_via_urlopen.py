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
def test_it_retrieves_local_refs_via_urlopen(self):
    with tempfile.NamedTemporaryFile(delete=False, mode='wt') as tempf:
        self.addCleanup(os.remove, tempf.name)
        json.dump({'foo': 'bar'}, tempf)
    ref = f'file://{pathname2url(tempf.name)}#foo'
    with self.resolver.resolving(ref) as resolved:
        self.assertEqual(resolved, 'bar')