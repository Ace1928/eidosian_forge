import io
import json
from itertools import islice
from typing import Any, Callable, Dict, List
import numpy as np
import pyarrow as pa
import datasets
def npz_loads(data: bytes):
    return np.load(io.BytesIO(data), allow_pickle=False)