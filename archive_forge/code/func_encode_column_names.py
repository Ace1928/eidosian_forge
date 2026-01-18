from datetime import date, datetime
from typing import Any, Dict, Iterable, Optional, Tuple
import numpy as np
import pandas as pd
import pyarrow as pa
from triad import Schema
from triad.utils.pyarrow import TRIAD_DEFAULT_TIMESTAMP
from triad.utils.schema import quote_name
def encode_column_names(names: Iterable[str]) -> Iterable[str]:
    for name in names:
        yield encode_column_name(name)