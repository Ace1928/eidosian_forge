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
def test_heterogeneous_unevaluatedItems_prefixItems(self):
    schema = {'prefixItems': [{}], 'unevaluatedItems': False}
    message = self.message_for(instance=['foo', 'bar', 37], schema=schema)
    self.assertEqual(message, "Unevaluated items are not allowed ('bar', 37 were unexpected)")