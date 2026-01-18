import hashlib
import torch
import torch.fx
from typing import Any, Dict, Optional, TYPE_CHECKING
from torch.fx.node import _get_qualified_name, _format_arg
from torch.fx.graph import _parse_stack_trace
from torch.fx.passes.shape_prop import TensorMetadata
from torch.fx._compatibility import compatibility
from itertools import chain
def get_dot_graph(self, submod_name=None) -> pydot.Dot:
    """
            Visualize a torch.fx.Graph with graphviz
            Example:
                >>> # xdoctest: +REQUIRES(module:pydot)
                >>> # define module
                >>> class MyModule(torch.nn.Module):
                >>>     def __init__(self):
                >>>         super().__init__()
                >>>         self.linear = torch.nn.Linear(4, 5)
                >>>     def forward(self, x):
                >>>         return self.linear(x).clamp(min=0.0, max=1.0)
                >>> module = MyModule()
                >>> # trace the module
                >>> symbolic_traced = torch.fx.symbolic_trace(module)
                >>> # setup output file
                >>> import ubelt as ub
                >>> dpath = ub.Path.appdir('torch/tests/FxGraphDrawer').ensuredir()
                >>> fpath = dpath / 'linear.svg'
                >>> # draw the graph
                >>> g = FxGraphDrawer(symbolic_traced, "linear")
                >>> g.get_dot_graph().write_svg(fpath)
            """
    if submod_name is None:
        return self.get_main_dot_graph()
    else:
        return self.get_submod_dot_graph(submod_name)