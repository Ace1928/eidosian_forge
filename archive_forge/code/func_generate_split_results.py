import argparse
import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple, Sequence, Iterable, Any, List, Dict, Optional, Tuple
import logging
import torch
from torch.fx.passes.graph_manipulation import get_size_of_node
from torch.fx.node import map_arg
from torch.fx._compatibility import compatibility
from .operator_support import (
from .graph_drawer import FxGraphDrawer
from .shape_prop import ShapeProp
from .split_utils import split_by_tags
from .tools_common import (
def generate_split_results(self) -> SplitResult:
    split_module = self()
    submodule_names = []
    for name, mod in split_module.named_children():
        submodule_names.append(name)
    submodule_inputs = generate_inputs_for_submodules(split_module, self.sample_input, submodule_names)
    return SplitResult(split_module, submodule_inputs, self.non_acc_submodule_name)