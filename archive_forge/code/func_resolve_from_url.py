from __future__ import division
import contextlib
import json
import numbers
from jsonschema import _utils, _validators
from jsonschema.compat import (
from jsonschema.exceptions import ErrorTree  # Backwards compat  # noqa: F401
from jsonschema.exceptions import RefResolutionError, SchemaError, UnknownType
def resolve_from_url(self, url):
    url, fragment = urldefrag(url)
    try:
        document = self.store[url]
    except KeyError:
        try:
            document = self.resolve_remote(url)
        except Exception as exc:
            raise RefResolutionError(exc)
    return self.resolve_fragment(document, fragment)