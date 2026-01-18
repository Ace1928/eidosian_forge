from __future__ import division
import contextlib
import json
import numbers
from jsonschema import _utils, _validators
from jsonschema.compat import (
from jsonschema.exceptions import ErrorTree  # Backwards compat  # noqa: F401
from jsonschema.exceptions import RefResolutionError, SchemaError, UnknownType
def pop_scope(self):
    try:
        self._scopes_stack.pop()
    except IndexError:
        raise RefResolutionError('Failed to pop the scope from an empty stack. `pop_scope()` should only be called once for every `push_scope()`')