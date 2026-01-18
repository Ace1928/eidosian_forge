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
def test_additionalProperties_false_patternProperties(self):
    schema = {'type': 'object', 'additionalProperties': False, 'patternProperties': {'^abc$': {'type': 'string'}, '^def$': {'type': 'string'}}}
    message = self.message_for(instance={'zebra': 123}, schema=schema, cls=validators.Draft4Validator)
    self.assertEqual(message, '{} does not match any of the regexes: {}, {}'.format(repr('zebra'), repr('^abc$'), repr('^def$')))
    message = self.message_for(instance={'zebra': 123, 'fish': 456}, schema=schema, cls=validators.Draft4Validator)
    self.assertEqual(message, '{}, {} do not match any of the regexes: {}, {}'.format(repr('fish'), repr('zebra'), repr('^abc$'), repr('^def$')))