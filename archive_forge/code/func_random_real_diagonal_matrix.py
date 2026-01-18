import random
from typing import Tuple, Optional
import numpy as np
import pytest
import cirq
def random_real_diagonal_matrix(n: int, d: Optional[int]=None) -> np.ndarray:
    return np.diag([random.random() if d is None or k < d else 0 for k in range(n)])