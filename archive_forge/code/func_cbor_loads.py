import io
import json
from itertools import islice
from typing import Any, Callable, Dict, List
import numpy as np
import pyarrow as pa
import datasets
def cbor_loads(data: bytes):
    import cbor
    return cbor.loads(data)