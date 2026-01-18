import collections
import copy
import functools
import itertools
import operator
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List
import torch
from transformers.file_utils import add_end_docstrings
from transformers.utils.fx import _gen_constructor_wrapper
def get_transformed_nodes(self, graph_module: 'GraphModule') -> List['Node']:
    """
        Args:
            graph_module (`torch.fx.GraphModule`):
                The graph_module to get the nodes from.

        Returns:
            `List[torch.fx.Node]`:
                Gives the list of nodes that were transformed by the transformation.
        """
    return [node for node in graph_module.graph.nodes if self.transformed(node)]