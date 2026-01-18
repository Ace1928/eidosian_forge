import random
from typing import Tuple, Optional
import numpy as np
import pytest
import cirq
def assert_bidiagonalized_by(m, p, q, rtol: float=1e-05, atol: float=1e-08):
    d = p.dot(m).dot(q)
    assert cirq.is_orthogonal(p) and cirq.is_orthogonal(q) and cirq.is_diagonal(d, atol=atol), _get_assert_bidiagonalized_by_str(m, p, q, d)