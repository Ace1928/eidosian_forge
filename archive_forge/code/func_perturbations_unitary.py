import itertools
import numpy as np
import pytest
import cirq
import sympy
def perturbations_unitary(u, amount=1e-10):
    """Returns several unitaries in the neighborhood of u to test for numerical
    corner cases near critical values."""
    kak = cirq.kak_decomposition(u)
    yield u
    for i in range(3):
        for neg in (-1, 1):
            perturb_xyz = list(kak.interaction_coefficients)
            perturb_xyz[i] += neg * amount
            yield cirq.unitary(cirq.KakDecomposition(global_phase=kak.global_phase, single_qubit_operations_before=kak.single_qubit_operations_before, single_qubit_operations_after=kak.single_qubit_operations_after, interaction_coefficients=tuple(perturb_xyz)))