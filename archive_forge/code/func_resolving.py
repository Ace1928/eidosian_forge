from __future__ import division
import contextlib
import json
import numbers
from jsonschema import _utils, _validators
from jsonschema.compat import (
from jsonschema.exceptions import ErrorTree  # Backwards compat  # noqa: F401
from jsonschema.exceptions import RefResolutionError, SchemaError, UnknownType
@contextlib.contextmanager
def resolving(self, ref):
    """
        Context manager which resolves a JSON ``ref`` and enters the
        resolution scope of this ref.

        Arguments:

            ref (str):

                The reference to resolve

        """
    url, resolved = self.resolve(ref)
    self.push_scope(url)
    try:
        yield resolved
    finally:
        self.pop_scope()