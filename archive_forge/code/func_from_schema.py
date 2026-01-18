from __future__ import division
import contextlib
import json
import numbers
from jsonschema import _utils, _validators
from jsonschema.compat import (
from jsonschema.exceptions import ErrorTree  # Backwards compat  # noqa: F401
from jsonschema.exceptions import RefResolutionError, SchemaError, UnknownType
@classmethod
def from_schema(cls, schema, *args, **kwargs):
    """
        Construct a resolver from a JSON schema object.

        Arguments:

            schema:

                the referring schema

        Returns:

            :class:`RefResolver`

        """
    return cls(schema.get(u'id', u''), schema, *args, **kwargs)