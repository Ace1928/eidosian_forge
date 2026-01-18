import itertools
import numpy as np
import pytest
import cirq
import sympy
def assert_specific_sqrt_iswap_count(operations, count):
    actual = sum((len(op.qubits) == 2 for op in operations))
    assert actual == count, f'Incorrect sqrt-iSWAP count.  Expected {count} but got {actual}.'