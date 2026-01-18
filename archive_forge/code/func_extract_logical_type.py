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
def extract_logical_type(schema: Schema) -> Optional[str]:
    if not isinstance(schema, dict):
        return None
    d_schema = schema
    rt = d_schema['type']
    lt = d_schema.get('logicalType')
    if lt:
        return f'{rt}-{lt}'
    return None