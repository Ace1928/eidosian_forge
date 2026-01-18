from typing import Optional, Sequence, Type
import pytest
import cirq
import sympy
import numpy as np
def assert_optimizes(before: cirq.Circuit, expected: cirq.Circuit, additional_gates: Optional[Sequence[Type[cirq.Gate]]]=None):
    if additional_gates is None:
        gateset = cirq.CZTargetGateset()
    else:
        gateset = cirq.CZTargetGateset(additional_gates=additional_gates)
    cirq.testing.assert_same_circuits(cirq.optimize_for_target_gateset(before, gateset=gateset, ignore_failures=False), expected)