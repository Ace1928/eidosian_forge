import collections
import contextlib
import copy
import inspect
import json
import sys
import textwrap
from typing import (
from itertools import zip_longest
from importlib.metadata import version as importlib_version
from typing import Final
import jsonschema
import jsonschema.exceptions
import jsonschema.validators
import numpy as np
import pandas as pd
from packaging.version import Version
from altair import vegalite
@classmethod
def resolve_references(cls, schema: Optional[dict]=None) -> dict:
    """Resolve references in the context of this object's schema or root schema."""
    schema_to_pass = schema or cls._schema
    assert schema_to_pass is not None
    return _resolve_references(schema=schema_to_pass, rootschema=cls._rootschema or cls._schema or schema)