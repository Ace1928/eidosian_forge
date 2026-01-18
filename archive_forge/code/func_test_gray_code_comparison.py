import functools
import itertools
import math
import random
import numpy as np
import pytest
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.ops.boolean_hamiltonian as bh
@pytest.mark.parametrize('seq_a,seq_b,expected', [((), (), 0), ((), (0,), -1), ((0,), (), 1), ((0,), (0,), 0)])
def test_gray_code_comparison(seq_a, seq_b, expected):
    assert bh._gray_code_comparator(seq_a, seq_b) == expected