from __future__ import annotations
from enum import IntEnum
from typing import Callable
import numpy as np
from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl
def build_node(current_node_index, is_leaf) -> Node | Leaf:
    if is_leaf:
        return Leaf(leaf_weights[current_node_index], leaf_targetids[current_node_index])
    if nodes_modes[current_node_index] == Mode.MEMBER:
        set_members = set()
        while (set_member := next(set_membership_iter)) and (not np.isnan(set_member)):
            set_members.add(set_member)
        node = Node(nodes_modes[current_node_index], set_members, nodes_featureids[current_node_index], nodes_missing_value_tracks_true[current_node_index] if nodes_missing_value_tracks_true is not None else False)
    else:
        node = Node(nodes_modes[current_node_index], nodes_splits[current_node_index], nodes_featureids[current_node_index], nodes_missing_value_tracks_true[current_node_index] if nodes_missing_value_tracks_true is not None else False)
    node.true_branch = build_node(nodes_truenodeids[current_node_index], nodes_trueleafs[current_node_index])
    node.false_branch = build_node(nodes_falsenodeids[current_node_index], nodes_falseleafs[current_node_index])
    return node