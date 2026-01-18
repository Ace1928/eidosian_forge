import collections
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.testing
from cirq.study.result import _pack_digits
def test_json_bit_packing_error():
    with pytest.raises(ValueError):
        _pack_digits(np.ones(10), pack_bits='hi mom')