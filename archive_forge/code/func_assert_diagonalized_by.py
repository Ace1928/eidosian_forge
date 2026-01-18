import random
from typing import Tuple, Optional
import numpy as np
import pytest
import cirq
def assert_diagonalized_by(m, p, atol: float=1e-08):
    d = p.T.dot(m).dot(p)
    assert cirq.is_orthogonal(p) and cirq.is_diagonal(d, atol=atol), _get_assert_diagonalized_by_str(m, p, d)