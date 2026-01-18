from typing import List, Sequence, Tuple
import itertools
import numpy as np
import pytest
import sympy
import cirq
def assert_decomposition_valid(cphase_gate, fsim_gate):
    u_expected = cirq.unitary(cphase_gate)
    ops = cirq.decompose_cphase_into_two_fsim(cphase_gate, fsim_gate=fsim_gate)
    u_actual = cirq.unitary(cirq.Circuit(ops))
    assert np.allclose(u_actual, u_expected, atol=1e-06)