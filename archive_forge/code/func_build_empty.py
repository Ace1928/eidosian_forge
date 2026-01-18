import pickle
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import pyarrow as pa
import ray
import ray.data as rd
from triad import Schema
from .._constants import _ZERO_COPY
def build_empty(schema: Schema) -> rd.Dataset:
    return rd.from_arrow(schema.create_empty_arrow_table())