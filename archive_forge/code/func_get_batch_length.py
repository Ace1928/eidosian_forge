import math
from typing import Any, Generic, Iterable, List, Optional, Tuple, TypeVar
import numpy as np
import pandas as pd
import pyarrow as pa
from triad.utils.convert import to_size
from .iter import slice_iterable
def get_batch_length(self, batch: pa.Table) -> int:
    return batch.num_rows