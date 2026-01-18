import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import re
import sys
from copy import copy, deepcopy
from typing import Dict, List, Optional, Set, Tuple, Union
import sympy
import torch
import torch.fx
from torch._inductor import dependencies
from torch._inductor.ir import StorageBox, TensorBox
from torch._prims_common import is_float_dtype
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges
from .. import codecache, config, ir, metrics
from ..codegen.wrapper import WrapperCodeGen
from ..optimize_indexing import range_expressable_in_32_bits
from ..scheduler import BaseScheduling, SchedulerNode
from ..utils import (
from ..virtualized import ops, V
from .common import (
def is_mask(self, name: str, users: Dict[torch.fx.Node, None]):
    load_type = V.graph.get_dtype(name)
    if load_type == torch.bool:
        return all((user.target in ('where', 'masked') for user in users.keys()))
    elif load_type == torch.uint8:
        '\n            If the load value is torch.uint8, then we only support the loaded\n            value is as the mask.\n            '
        if not all((user.target == 'to_dtype' and user.args[-1] == torch.bool for user in users.keys())):
            return False
        for to_dtype_node in users.keys():
            assert to_dtype_node.target == 'to_dtype'
            if not all((user.target in ('where', 'masked') for user in to_dtype_node.users.keys())):
                return False
        return True
    else:
        return False