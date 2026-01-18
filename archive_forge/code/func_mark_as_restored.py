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
def mark_as_restored(self, node: 'Node'):
    """
        Marks a node as restored back to its original state.

        Args:
            node (`torch.fx.Node`):
                The node to mark as restored.
        """
    node_transformations = getattr(node, 'transformations', set())
    if self.signature not in node_transformations:
        raise ValueError('The node was not transformed by this transformation.')
    node_transformations.remove(self.signature)