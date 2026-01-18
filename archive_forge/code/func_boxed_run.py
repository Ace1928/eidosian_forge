from .graph_module import GraphModule
from .graph import Graph
from .node import Argument, Node, Target, map_arg, map_aggregate
from .proxy import Proxy
from ._symbolic_trace import Tracer
from ._compatibility import compatibility
from . import config
import torch.fx.traceback as fx_traceback
import torch
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import inspect
from contextlib import contextmanager
from torch.hub import tqdm
@compatibility(is_backward_compatible=True)
def boxed_run(self, args_list):
    """
        Run `module` via interpretation and return the result.  This uses the "boxed"
        calling convention, where you pass a list of arguments, which will be cleared
        by the interpreter.  This ensures that input tensors are promptly deallocated.
        """
    args_iter = iter(args_list)
    env = {}
    for n in self.module.graph.nodes:
        if n.op == 'placeholder':
            env[n] = next(args_iter)
    args_list.clear()
    return self.run(initial_env=env)