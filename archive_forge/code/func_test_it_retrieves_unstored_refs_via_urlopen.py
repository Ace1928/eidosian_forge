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
def test_it_retrieves_unstored_refs_via_urlopen(self):
    ref = 'http://bar#baz'
    schema = {'baz': 12}
    if 'requests' in sys.modules:
        self.addCleanup(sys.modules.__setitem__, 'requests', sys.modules['requests'])
    sys.modules['requests'] = None

    @contextmanager
    def fake_urlopen(url):
        self.assertEqual(url, 'http://bar')
        yield BytesIO(json.dumps(schema).encode('utf8'))
    self.addCleanup(setattr, validators, 'urlopen', validators.urlopen)
    validators.urlopen = fake_urlopen
    with self.resolver.resolving(ref) as resolved:
        pass
    self.assertEqual(resolved, 12)