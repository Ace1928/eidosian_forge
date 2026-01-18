from __future__ import annotations
import itertools
import logging
import weakref
from typing import Any, List, Optional, Tuple
import torch
import torch.utils._pytree as pytree
from torch._dynamo.utils import dynamo_timed, lazy_format_graph_code
from torch._functorch.aot_autograd import MutationType
from torch._functorch.compile_utils import fx_graph_cse
from torch._inductor.constant_folding import constant_fold, replace_node_with_constant
from torch._inductor.fx_passes.freezing_patterns import freezing_passes
from torch._inductor.fx_passes.post_grad import view_to_reshape
from . import config
def enforce_output_layout(gm: torch.fx.GraphModule):
    """
    Make sure the output node's layout does not change due to compiler optimizations
    by adding aten.as_strided nodes with the expected strides.

    Only used for inference so we can assume all graph outputs are model outputs.
    """
    *_, output_node = gm.graph.nodes
    out_list = output_node.args[0]
    with gm.graph.inserting_before(output_node):
        for n in out_list:
            if not isinstance(n.meta['val'], torch.Tensor) or not torch._prims_common.is_non_overlapping_and_dense(n.meta['val']):
                continue
            ft = n.meta['val']
            new_node = gm.graph.call_function(prims.inductor_force_stride_order.default, (n, ft.stride()))
            output_node.replace_input_with(n, new_node)
    gm.graph.lint()
    gm.recompile()