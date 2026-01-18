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
def replace_params_with_constants(gm: torch.fx.GraphModule, flat_params: list[Any], fw_metadata: torch._functorch.aot_autograd.ViewAndMutationMeta) -> List[int]:
    """
    Replaces the parameters of a PyTorch GraphModule with constants wherever possible.
    Returns a list of indices representing the input parameters that were not converted to constants.
    """
    params = [node for node in gm.graph.nodes if node.op == 'placeholder']
    fake_inp_nodes = params[:len(params)]
    preserved_arg_indices = []
    aliased_input_args = [out_info.base_idx for out_info in fw_metadata.output_info if out_info.base_idx is not None]
    mutated_inps = [i for i, m in enumerate(fw_metadata.input_info) if m.mutation_type in (MutationType.MUTATED_IN_GRAPH, MutationType.MUTATED_OUT_GRAPH)]
    for i, (real_input, node) in enumerate(zip(flat_params, fake_inp_nodes)):
        if i in mutated_inps or i in aliased_input_args:
            preserved_arg_indices.append(i)
            continue
        replace_node_with_constant(gm, node, real_input)
    preserved_arg_indices.extend(range(len(flat_params), len(params)))
    gm.recompile()
    return preserved_arg_indices