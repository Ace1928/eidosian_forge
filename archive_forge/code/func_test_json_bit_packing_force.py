import collections
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.testing
from cirq.study.result import _pack_digits
def test_json_bit_packing_force():
    assert _pack_digits(np.ones(10, dtype=int), pack_bits='force') == _pack_digits(np.ones(10), pack_bits='auto')
    assert _pack_digits(2 * np.ones(10, dtype=int), pack_bits='force') != _pack_digits(2 * np.ones(10, dtype=int), pack_bits='auto')
    assert _pack_digits(2 * np.ones(10, dtype=int), pack_bits='force') == _pack_digits(np.ones(10), pack_bits='auto')