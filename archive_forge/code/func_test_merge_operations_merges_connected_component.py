from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def test_merge_operations_merges_connected_component():
    c_orig = _create_circuit_to_merge()
    cirq.testing.assert_has_diagram(c_orig, "\n0: ───H───@───@───H───@───X───────@───────X───X['ignore']───@───\n          │   │       │           │                         │\n1: ───H───┼───X───────@───────Y───X───@───────Y─────────────X───\n          │                           │\n2: ───H───X───────────────────────────X─────────────────────────\n")

    def merge_func(op1, op2):
        """Artificial example where a CZ will absorb any merge-able operation."""
        for op in [op1, op2]:
            if op.gate == cirq.CZ:
                return op
        return None
    c_new = cirq.merge_operations(c_orig, merge_func)
    cirq.testing.assert_has_diagram(c_new, '\n0: ───H───@───────────@───────────────────────────@───\n          │           │                           │\n1: ───────┼───────────@───────────────@───────Y───X───\n          │                           │\n2: ───H───X───────────────────────────X───────────────')