import builtins
import collections
import functools
import inspect
import math
import operator
import os
import random
import warnings
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch import nn
from torch.fx import Graph, GraphModule, Proxy, Tracer
from torch.fx._compatibility import compatibility
from torch.fx.proxy import ParameterProxy
from .. import PretrainedConfig, PreTrainedModel, logging
from ..models.auto import get_values
from ..models.auto.modeling_auto import (
from ..pytorch_utils import is_torch_greater_or_equal_than_2_0
from ..utils import (
def path_of_module(self, mod: nn.Module) -> str:
    """
        Helper method to find the qualified name of `mod` in the Module hierarchy of `root`. For example, if `root` has
        a submodule named `foo`, which has a submodule named `bar`, passing `bar` into this function will return the
        string "foo.bar".

        Args:
            mod (str): The `Module` to retrieve the qualified name for.
        """
    try:
        return super().path_of_module(mod)
    except NameError as e:
        if self.allow_insert_stateless_mods and len(list(mod.parameters())) == 0 and (len(list(mod.buffers())) == 0):
            path = self._insert_module_as_submodule(mod)
            return path
        raise e