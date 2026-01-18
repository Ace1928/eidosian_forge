from __future__ import division
import contextlib
import json
import numbers
from jsonschema import _utils, _validators
from jsonschema.compat import (
from jsonschema.exceptions import ErrorTree  # Backwards compat  # noqa: F401
from jsonschema.exceptions import RefResolutionError, SchemaError, UnknownType
@contextlib.contextmanager
def in_scope(self, scope):
    self.push_scope(scope)
    try:
        yield
    finally:
        self.pop_scope()