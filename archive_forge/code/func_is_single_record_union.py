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
def is_single_record_union(schema: List[Schema]) -> bool:
    return _get_name_and_record_counts_from_union(schema)[1] == 1