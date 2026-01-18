from typing import Optional, Sequence, Type
import pytest
import cirq
import sympy
import numpy as np
def assert_optimization_not_broken(circuit: cirq.Circuit):
    c_new = cirq.optimize_for_target_gateset(circuit, gateset=cirq.CZTargetGateset())
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(circuit, c_new, atol=1e-06)
    c_new = cirq.optimize_for_target_gateset(circuit, gateset=cirq.CZTargetGateset(allow_partial_czs=True), ignore_failures=False)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(circuit, c_new, atol=1e-06)