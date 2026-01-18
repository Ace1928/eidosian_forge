from typing import Any, cast, Iterable, Optional, Tuple
import numpy as np
import pytest
import cirq
def make_buffers(shape, dtype):
    return (np.empty(shape, dtype=dtype), np.empty(shape, dtype=dtype), np.empty(shape, dtype=dtype))