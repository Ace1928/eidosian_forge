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
def get_submod_inputs(main_mod, submod, example_inputs):
    sub_inputs = None

    def get_inputs(self, inputs):
        nonlocal sub_inputs
        sub_inputs = inputs
    handle = submod.register_forward_pre_hook(get_inputs)
    main_mod(*example_inputs)
    handle.remove()
    return sub_inputs