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
@dynamo_timed
def convert_conv_weights_to_channels_last(gm: torch.fx.GraphModule):
    """
    Convert 4d convolution weight tensor to channels last format.

    This pass is performed before freezing so the added nodes can be constant
    folded by freezing.
    """
    convs = [n for n in gm.graph.nodes if n.target == aten.convolution.default]
    for conv in convs:
        weight_node = conv.args[1]
        if len(weight_node.meta['val'].size()) != 4 or weight_node.meta['val'].is_contiguous(memory_format=torch.channels_last):
            continue
        with gm.graph.inserting_before(conv):
            new_node = gm.graph.call_function(aten.clone.default, (weight_node,), {'memory_format': torch.channels_last})
            conv.replace_input_with(weight_node, new_node)
    enforce_as_strided_input_layout(gm)
    enforce_output_layout(gm)