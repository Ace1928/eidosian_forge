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
def test_maxProperties(self):
    message = self.message_for(instance={'a': {}, 'b': {}, 'c': {}}, schema={'maxProperties': 2})
    self.assertEqual(message, "{'a': {}, 'b': {}, 'c': {}} has too many properties")