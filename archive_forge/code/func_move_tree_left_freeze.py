from typing import cast
import pytest
import cirq
def move_tree_left_freeze(root):
    return cirq.freeze_op_tree(cirq.transform_op_tree(root, move_left))