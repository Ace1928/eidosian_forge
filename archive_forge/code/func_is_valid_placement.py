import abc
import warnings
from dataclasses import dataclass
from typing import (
import networkx as nx
from matplotlib import pyplot as plt
from cirq import _compat
from cirq.devices import GridQubit, LineQubit
from cirq.protocols.json_serialization import dataclass_json_dict
def is_valid_placement(big_graph: nx.Graph, small_graph: nx.Graph, small_to_big_mapping: Dict):
    """Return whether the given placement is a valid placement of small_graph onto big_graph.

    This is done by making sure all the nodes and edges on the mapped version of `small_graph`
    are present in `big_graph`.

    Args:
        big_graph: A larger graph we're placing `small_graph` onto.
        small_graph: A smaller, (potential) sub-graph to validate the given mapping.
        small_to_big_mapping: A mapping from `small_graph` nodes to `big_graph`
            nodes. After the mapping occurs, we check whether all of the mapped nodes and
            edges exist on `big_graph`.
    """
    small_mapped = nx.relabel_nodes(small_graph, small_to_big_mapping)
    return _is_valid_placement_helper(big_graph=big_graph, small_mapped=small_mapped, small_to_big_mapping=small_to_big_mapping)