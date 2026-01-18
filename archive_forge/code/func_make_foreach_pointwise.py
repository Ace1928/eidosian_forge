import functools
import itertools
import logging
import os
import warnings
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import sympy
import torch
import torch.fx
import torch.utils._pytree as pytree
from torch._higher_order_ops.triton_kernel_wrap import (
from torch._prims_common import (
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch.utils._sympy.functions import CeilDiv, FloorDiv, ModularIndexing
from .._dynamo.utils import import_submodule
from . import config, inductor_prims, ir, test_operators  # NOQA: F401
from .decomposition import decompositions, get_decompositions
from .ir import (
from .utils import (
from .virtualized import ops, V
from . import kernel
import_submodule(kernel)
from . import quantized_lowerings
def make_foreach_pointwise(pw_fn, allow_alpha=False):

    def inner(*inputs: List[List[TensorBox]], alpha=1):

        def group_args(arg_pairs):
            out = defaultdict(list)
            for i, args in enumerate(arg_pairs):
                use_foreach = not is_dynamic(*args)
                device = None
                for t in args:
                    if isinstance(t, TensorBox):
                        device = t.data.get_device()
                        break
                assert device is not None, 'foreach op should have at least one tensor arg'
                out[device, use_foreach].append((i, args))
            return out
        realize_outputs = len(V.graph.current_node.users) == 0 or V.graph.current_node.target in inplace_foreach_ops
        for node in V.graph.current_node.users:
            for user in node.users:
                if not (user.op == 'call_function' and user.target in foreach_ops):
                    realize_outputs = True
        a_list_input = None
        for input in inputs:
            if isinstance(input, (list, tuple)):
                a_list_input = input
                break
        assert a_list_input is not None, 'at least one input must be a list to a foreach op'
        broadcast_inputs = []
        for input in inputs:
            if not isinstance(input, (list, tuple)):
                broadcast_inputs.append([input] * len(a_list_input))
            else:
                broadcast_inputs.append(input)
        groups = group_args(zip(*broadcast_inputs))
        outputs = [None] * len(a_list_input)
        for (device, use_foreach), group in groups.items():
            buffer_list = []
            for output_ind, args in group:
                if allow_alpha:
                    output = pw_fn(*args, alpha=alpha)
                else:
                    output = pw_fn(*args)
                outputs[output_ind] = output
                if device.type == 'cuda' and use_foreach and realize_outputs:
                    buffer_list.append(output.realize())
            if buffer_list:
                V.graph.register_list(buffer_list)
        assert all((x is not None for x in outputs))
        return outputs
    return inner