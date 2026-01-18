from typing import Tuple, List, cast
import re
import pytest
import sympy
import cirq
from cirq._compat import proper_repr
import numpy as np
def assert_validate_and_contains_consistent(gateset, op_tree, result):
    assert all((op in gateset for op in cirq.flatten_to_ops(op_tree))) is result
    for item in optree_and_circuit(op_tree):
        assert gateset.validate(item) is result