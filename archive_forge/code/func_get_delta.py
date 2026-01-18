from __future__ import annotations
import itertools
import operator
from typing import TYPE_CHECKING
import networkx as nx
import numpy as np
from monty.json import MSONable
def get_delta(node1, node2, edge_data):
    """
    Get the delta.

    Args:
        node1:
        node2:
        edge_data:
    """
    if node1.isite == edge_data['start'] and node2.isite == edge_data['end']:
        return np.array(edge_data['delta'], dtype=int)
    if node2.isite == edge_data['start'] and node1.isite == edge_data['end']:
        return -np.array(edge_data['delta'], dtype=int)
    raise ValueError('Trying to find a delta between two nodes with an edge that seems not to link these nodes.')