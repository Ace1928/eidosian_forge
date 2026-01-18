from typing import TYPE_CHECKING, Union, Callable, Any, Tuple, List, Optional, Dict, Set
from ._compatibility import compatibility
from .immutable_collections import immutable_dict, immutable_list
import torch
import builtins
import types
import inspect
import warnings
from torch.fx.operator_schemas import normalize_function, normalize_module, ArgsKwargsPair
from .._ops import ops as _ops
@property
def all_input_nodes(self) -> List['Node']:
    """
        Return all Nodes that are inputs to this Node. This is equivalent to
        iterating over ``args`` and ``kwargs`` and only collecting the values that
        are Nodes.

        Returns:

            List of ``Nodes`` that appear in the ``args`` and ``kwargs`` of this
            ``Node``, in that order.
        """
    return list(self._input_nodes.keys())