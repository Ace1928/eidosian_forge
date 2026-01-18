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
@torch.utils._python_dispatch._disable_current_modes()
def discard_traced_gm_params(mod: torch.fx.GraphModule):
    for attr_name, tensor in list(itertools.chain(mod.named_parameters(recurse=False), mod.named_buffers(recurse=False))):
        with torch._dispatch.python.no_python_dispatcher():
            e_t = ErasedTensor(tensor, attr_name, mod)
        if isinstance(tensor, torch.nn.Parameter):
            e_t.requires_grad_(True)
            e_t._is_param = True
        setattr(mod, attr_name, e_t)