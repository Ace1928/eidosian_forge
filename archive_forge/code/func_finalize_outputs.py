import copy
import operator
from copy import deepcopy
from typing import cast, Dict, List, Optional, Union
import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.export import ExportedProgram
from torch.export.exported_program import (
from torch.fx import GraphModule
from .utils import _check_input_constraints_pre_hook
def finalize_outputs(self):
    orig_outputs = []
    signature = self.module_call_graph.get(self.fqn)
    if signature is not None and self.parent is not None:
        for output in signature.outputs:
            if isinstance(output, (TensorArgument, SymIntArgument)):
                orig_outputs.append(self.seen_nodes[output.name])
            else:
                raise RuntimeError(f'Unsupported data type for output node: {output}')
        tree_out_node = _generate_unflatten(self.graph_module, tuple((self.node_map[self.seen_nodes[output.name]] for output in orig_outputs)), signature.out_spec)
        parent_out: Optional[torch.fx.Node] = _generate_flatten(self.parent.graph_module, self.parent_call_module, signature.out_spec)
        graph_outputs: Union[torch.fx.Node, List[torch.fx.Node]] = tree_out_node
    else:
        graph_outputs = []
        for orig_node in self.node_map.keys():
            for user_node in orig_node.users:
                if user_node.name not in self.seen_nodes:
                    orig_outputs.append(orig_node)
                    graph_outputs.append(self.node_map[orig_node])
                    break
        parent_out = self.parent_call_module
        if len(graph_outputs) == 1:
            graph_outputs = graph_outputs[0]
    assert isinstance(graph_outputs, (list, torch.fx.Node))
    self.graph.output(graph_outputs)
    self.graph.lint()
    self.graph_module.recompile()
    if parent_out is None:
        return
    if len(orig_outputs) == 1 and signature is None:
        self.parent.node_map[orig_outputs[0]] = parent_out
    else:
        for i, orig_output in enumerate(orig_outputs):
            proxy_out = torch.fx.Proxy(parent_out)[i].node
            self.parent.node_map[orig_output] = proxy_out
    if self.cached_graph_module is not None:
        _verify_graph_equivalence(self.cached_graph_module, self.graph_module)