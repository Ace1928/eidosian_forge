import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, List, Optional
import torch
from torch import fx
from torch._dynamo.output_graph import GraphCompileReason
from torch._dynamo.utils import deepcopy_to_fake_tensor, detect_fake_mode
from torch.fx.node import Node
def compile_submod(self, input_mod, args, kwargs):
    """
                Compile the submodule,
                using a wrapper to make sure its output is always a tuple,
                which is required by AotAutograd based compilers
                """
    assert len(kwargs) == 0, 'We assume only args for these modules'

    class WrapperModule(torch.nn.Module):

        def __init__(self, submod, unwrap_singleton_tuple):
            super().__init__()
            self.submod = submod
            self.unwrap_singleton_tuple = unwrap_singleton_tuple

        def forward(self, *args):
            x = self.submod(*args)
            if self.unwrap_singleton_tuple and isinstance(x, (tuple, list)):
                return x[0]
            return x
    unwrap_singleton_tuple = False
    for sn in input_mod.graph.nodes:
        if sn.op == 'output':
            if not isinstance(sn.args[0], tuple):
                unwrap_singleton_tuple = True
                sn.args = (sn.args,)
    input_mod.recompile()
    input_mod.compile_subgraph_reason = GraphCompileReason('DDPOptimizer intentional graph-break (See Note [DDPOptimizer]). Set `torch._dynamo.config.optimize_ddp = False` to disable.', [traceback.FrameSummary(__file__, 0, DDPOptimizer)])
    wrapper = WrapperModule(self.compiler(input_mod, args), unwrap_singleton_tuple)
    return wrapper