from __future__ import annotations
import itertools
import operator
from typing import TYPE_CHECKING
import networkx as nx
import numpy as np
from monty.json import MSONable
@classmethod
def from_edges(cls, edges, edges_are_ordered: bool=True) -> Self:
    """Constructs SimpleGraphCycle from a list edges.

        By default, the edges list is supposed to be ordered as it will be
        much faster to construct the cycle. If edges_are_ordered is set to
        False, the code will automatically try to find the corresponding edge
        order in the list.
        """
    if edges_are_ordered:
        nodes = [edge[0] for edge in edges]
        if not all((e1e2[0][1] == e1e2[1][0] for e1e2 in zip(edges, edges[1:]))) or edges[-1][1] != edges[0][0]:
            raise ValueError('Could not construct a cycle from edges.')
    else:
        remaining_edges = list(edges)
        nodes = list(remaining_edges.pop())
        while remaining_edges:
            prev_node = nodes[-1]
            for ie, e in enumerate(remaining_edges):
                if prev_node == e[0]:
                    remaining_edges.pop(ie)
                    nodes.append(e[1])
                    break
                if prev_node == e[1]:
                    remaining_edges.pop(ie)
                    nodes.append(e[0])
                    break
            else:
                raise ValueError('Could not construct a cycle from edges.')
        if nodes[0] != nodes[-1]:
            raise ValueError('Could not construct a cycle from edges.')
        nodes.pop()
    return cls(nodes)