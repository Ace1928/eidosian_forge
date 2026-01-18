from typing import cast, Type
from unittest import mock
import numpy as np
import pytest
import cirq
def assert_not_affected(state: np.ndarray, sample: float):
    np.testing.assert_allclose(get_result(state, sample), state, atol=1e-08)