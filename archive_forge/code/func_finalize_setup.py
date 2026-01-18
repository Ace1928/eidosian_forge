import copy
import inspect
import logging
from typing import Any, Callable, cast, Dict, List, Optional, Set, Tuple, Type
import torch.nn as nn
from torch import fx
from torch.distributed._spmd.graph_utils import (
from torch.distributed._spmd.partial_lower import partial_lower
from torch.fx.graph import _PyTreeCodeGen, PythonCode
from torch.fx.node import Argument
from torch.profiler import record_function
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_map, tree_map_only, tree_unflatten
def finalize_setup(self) -> None:
    """Set up the internal states and also get the signal from users that what is the maximum iteration count.

        This method must be called before the forward() is called.
        """
    if not self._is_frozen:
        self.graph.freeze_cross_iter_movement()
        self._num_extra_output = self.graph.num_extra_output
        if self._enable_inductor:
            self.main_gm = partial_lower(self.main_gm)
        self._is_frozen = True
    self._iter = 0