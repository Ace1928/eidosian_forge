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
def extract_record_type(schema: Schema) -> str:
    if isinstance(schema, dict):
        return schema['type']
    if isinstance(schema, list):
        return 'union'
    return schema