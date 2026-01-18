import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_conjugated_by_composite_with_disjoint_sub_gates():
    a, b = cirq.LineQubit.range(2)

    class DecomposeDisjoint(cirq.Gate):

        def num_qubits(self):
            return 2

        def _decompose_(self, qubits):
            yield cirq.H(qubits[1])
    assert cirq.X(a).conjugated_by(DecomposeDisjoint().on(a, b)) == cirq.X(a)
    assert cirq.X(a).pass_operations_over([DecomposeDisjoint().on(a, b)]) == cirq.X(a)