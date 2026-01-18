import hashlib
from io import StringIO
import math
from os import path
from copy import deepcopy
import re
from typing import Tuple, Set, Optional, List, Any
from .types import DictSchema, Schema, NamedSchemas
from .repository import (
from .const import AVRO_TYPES
from ._schema_common import (
def load_schema_ordered(ordered_schemas: List[str], *, _write_hint: bool=True) -> Schema:
    """Returns a schema loaded from a list of schemas.

    The list of schemas should be ordered such that any dependencies are listed
    before any other schemas that use those dependencies. For example, if schema
    `A` depends on schema `B` and schema B depends on schema `C`, then the list
    of schemas should be [C, B, A].

    Parameters
    ----------
    ordered_schemas
        List of paths to schemas
    _write_hint
        Internal API argument specifying whether or not the __fastavro_parsed
        marker should be added to the schema


    Consider the following example...


    Parent.avsc::

        {
            "type": "record",
            "name": "Parent",
            "namespace": "namespace",
            "fields": [
                {
                    "name": "child",
                    "type": "Child"
                }
            ]
        }


    namespace.Child.avsc::

        {
            "type": "record",
            "namespace": "namespace",
            "name": "Child",
            "fields": []
        }


    Code::

        from fastavro.schema import load_schema_ordered

        parsed_schema = load_schema_ordered(
            ["path/to/namespace.Child.avsc", "path/to/Parent.avsc"]
        )
    """
    loaded_schemas = []
    named_schemas: NamedSchemas = {}
    for idx, schema_path in enumerate(ordered_schemas):
        _last = _write_hint if idx + 1 == len(ordered_schemas) else False
        schema = load_schema(schema_path, named_schemas=named_schemas, _write_hint=_last)
        loaded_schemas.append(schema)
    top_first_order = loaded_schemas[::-1]
    outer_schema = top_first_order.pop(0)
    while top_first_order:
        sub_schema = top_first_order.pop(0)
        _inject_schema(outer_schema, sub_schema)
    return outer_schema