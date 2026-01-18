from dataclasses import dataclass, field
from torch.fx.graph import Graph
from torch.fx.node import Node
from torch.fx._compatibility import compatibility
from typing import Dict, List, Any, Type, Optional, Callable
import logging
import os
@compatibility(is_backward_compatible=False)
def check_subgraphs_connected(subgraph1: SourcePartition, subgraph2: SourcePartition) -> bool:
    """
    Given two subgraphs A and B (in the form of a list of nodes), checks if
    A has nodes connecting to at least one node in B -- aka there exists a node
    in B that uses a node in A (not the other way around).
    """
    for node in reversed(subgraph1.nodes):
        for user in node.users.keys():
            if user in subgraph2.nodes:
                return True
    return False