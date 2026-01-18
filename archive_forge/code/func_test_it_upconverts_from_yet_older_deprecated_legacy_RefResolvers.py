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
def test_it_upconverts_from_yet_older_deprecated_legacy_RefResolvers(self):
    """
        Legacy RefResolvers support only the context manager form of
        resolution.
        """

    class LegacyRefResolver:

        @contextmanager
        def resolving(this, ref):
            self.assertEqual(ref, 'the ref')
            yield {'type': 'integer'}
    resolver = LegacyRefResolver()
    schema = {'$ref': 'the ref'}
    with self.assertRaises(exceptions.ValidationError):
        self.Validator(schema, resolver=resolver).validate(None)