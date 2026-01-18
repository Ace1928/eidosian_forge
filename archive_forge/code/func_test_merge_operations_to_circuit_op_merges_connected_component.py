from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def test_merge_operations_to_circuit_op_merges_connected_component():
    c_orig = _create_circuit_to_merge()
    cirq.testing.assert_has_diagram(c_orig, "\n0: ───H───@───@───H───@───X───────@───────X───X['ignore']───@───\n          │   │       │           │                         │\n1: ───H───┼───X───────@───────Y───X───@───────Y─────────────X───\n          │                           │\n2: ───H───X───────────────────────────X─────────────────────────\n")

    def can_merge(ops1: List['cirq.Operation'], ops2: List['cirq.Operation']) -> bool:
        """Artificial example where a CZ will absorb any merge-able operation."""
        return any((o.gate == cirq.CZ for op_list in [ops1, ops2] for o in op_list))
    c_new = cirq.merge_operations_to_circuit_op(c_orig, can_merge, merged_circuit_op_tag='merged', tags_to_ignore=['ignore'])
    cirq.testing.assert_has_diagram(c_new, "\n                      [ 0: ───────@───H───@───X───@───X─── ]\n0: ───H───@───────────[           │       │       │        ]─────────────────────────────────X['ignore']───@───\n          │           [ 1: ───H───X───────@───Y───X─────── ]['merged']                                     │\n          │           │                                                                                    │\n1: ───────┼───────────#2─────────────────────────────────────────────────────────────@───────Y─────────────X───\n          │                                                                          │\n2: ───H───X──────────────────────────────────────────────────────────────────────────X─────────────────────────\n")