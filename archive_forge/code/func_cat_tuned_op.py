import functools
import itertools
import logging
import operator
from collections import Counter, defaultdict, namedtuple
from typing import Any, Dict, List, Optional, Set, Union
from sympy import Expr
import torch
import torch._inductor as inductor
import torch.utils._pytree as pytree
from torch import fx
from torch._decomp import register_decomposition
from torch._higher_order_ops.triton_kernel_wrap import triton_kernel_wrapper_functional
from torch._prims_common import is_boolean_dtype, is_expandable_to, is_integer_dtype
from torch._utils_internal import print_graph
from torch.fx.experimental.symbolic_shapes import definitely_true, sym_eq
from torch.fx.immutable_collections import immutable_dict
from .. import config, inductor_prims, ir, pattern_matcher
from ..fx_utils import FakeTensorUpdater, get_fake_args_kwargs, get_node_storage
from ..lowering import (
from ..pattern_matcher import (
from ..utils import decode_device, is_pointwise_use
from ..virtualized import V
from .group_batch_fusion import group_batch_fusion_passes
def cat_tuned_op(match, inputs, dim, *, op, shape_of):
    """
    Memory planning to remove cat. We can't use the stock memory
    planner since autotuning matmuls needs to know the output layout.
    """
    if len(inputs) == 1:
        return op(*inputs[0])
    if dim < 0:
        dim += len(shape_of(*inputs[0]))
    assert dim in (0, 1)
    notdim = 1 - dim
    new_size: Optional[Union[List[Expr], List[int]]] = None
    offsets_start = []
    offsets_end = []
    for i in range(len(inputs)):
        shape = shape_of(*inputs[i])
        if new_size is None:
            new_size = shape
        else:
            new_size[notdim] = V.graph.sizevars.guard_equals(shape[notdim], new_size[notdim])
            new_size[dim] += shape[dim]
        offsets_start.append(new_size[dim] - shape[dim])
        offsets_end.append(new_size[dim])
    assert new_size is not None
    dtype = functools.reduce(torch.promote_types, [x.get_dtype() for x in itertools.chain(*inputs)])
    device = inputs[0][0].get_device()
    kernel = ir.ConcatKernel(name=None, layout=ir.FixedLayout(device, dtype, new_size), inputs=[])
    kernel_tensor = ir.TensorBox.create(kernel)
    for i in range(len(inputs)):
        dst = ir.SliceView.create(kernel_tensor, dim, offsets_start[i], offsets_end[i])
        src = op(*inputs[i], layout=dst.get_layout()).data.data
        assert isinstance(src, (ir.ExternKernelOut, ir.TemplateBuffer))
        src.layout = ir.AliasedLayout(dst)
        kernel.inputs.append(src)
    kernel.name = V.graph.register_buffer(kernel)
    kernel.inputs = ir.ConcatKernel.unwrap_storage(kernel.inputs)
    return kernel_tensor