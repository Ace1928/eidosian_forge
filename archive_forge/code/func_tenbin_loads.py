import io
import json
from itertools import islice
from typing import Any, Callable, Dict, List
import numpy as np
import pyarrow as pa
import datasets
def tenbin_loads(data: bytes):
    from . import _tenbin
    return _tenbin.decode_buffer(data)