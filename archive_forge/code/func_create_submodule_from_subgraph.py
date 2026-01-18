import torch
import torch.fx
from torch.fx import (
from torch.ao.ns.fx.utils import (
from torch.ao.ns.fx.ns_types import (
from torch.ao.ns.fx.graph_passes import _maybe_get_fqn
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.utils import getattr_from_fqn
from torch.ao.quantization.fx.match_utils import _MatchResult
from torch.utils._pytree import tree_map
import collections
import copy
from typing import List, Dict, Set, Tuple, Callable, Any, Optional
import operator
def create_submodule_from_subgraph(model: torch.nn.Module, first_node: Node, last_node: Node) -> GraphModule:
    """
    Input: a model, and a linear subgraph within the model from first_node to
      last_node.

    Output: a new submodule containing a copy of the subgraph, with the inputs
      to the first node becoming the inputs to the submodule, and all other
      nodes in the subgraph being copied.

    Example inputs:

    `model`: a module with graph

      x0 -> op1 -> x1 -> op2 -> x2
             |
            arg1

    `first_node`: op1
    `last_node`: op2

    Example output: a new module with graph

      input1 -> op1_copy -> x1 -> op2_copy -> output1
                   |
                  arg1
    """

    class M(torch.nn.Module):

        def forward(self, x):
            pass
    m = M()
    gm = torch.fx.symbolic_trace(m)
    g = gm.graph
    for node in reversed(gm.graph.nodes):
        g.erase_node(node)
    cur_node_orig = first_node
    cur_args_orig = cur_node_orig.args
    cur_kwargs_orig = cur_node_orig.kwargs
    cur_name_idx = 0
    iteration_limit = 100
    cur_iteration = 0
    while True:
        if cur_node_orig is first_node:
            cur_args_copy = []
            cur_kwargs_copy = {}
            seen_names: Set[str] = set()
            old_name_to_new_node: Dict[str, Node] = {}

            def _add_placeholder(g: Graph, node: Node, seen_names, old_name_to_new_node):
                counter = 0
                while node.name + '_' + str(counter) in seen_names:
                    counter += 1
                cur_name = node.name + '_' + str(counter)
                seen_names.add(cur_name)
                placeholder = g.placeholder(cur_name)
                old_name_to_new_node[node.name] = placeholder
                return placeholder
            for arg in cur_node_orig.args:
                if isinstance(arg, Node):
                    p = _add_placeholder(g, arg, seen_names, old_name_to_new_node)
                    cur_args_copy.append(p)
                elif isinstance(arg, (list, tuple)):
                    new_arg = []
                    for inner_arg in arg:
                        if isinstance(inner_arg, Node):
                            new_arg.append(_add_placeholder(g, inner_arg, seen_names, old_name_to_new_node))
                        else:
                            new_arg.append(inner_arg)
                    cur_args_copy.append(new_arg)
                else:
                    cur_args_copy.append(arg)
            for kwarg_name, kwarg in cur_node_orig.kwargs.items():
                if isinstance(kwarg, Node):
                    cur_kwargs_copy[kwarg_name] = _add_placeholder(g, kwarg, seen_names, old_name_to_new_node)
                elif isinstance(kwarg, (list, tuple)):
                    new_kwarg = []
                    for inner_kwarg in kwarg:
                        p = _add_placeholder(g, inner_kwarg, seen_names, old_name_to_new_node)
                        new_kwarg.append(p)
                    cur_kwargs_copy[kwarg_name] = new_kwarg
                else:
                    cur_kwargs_copy[kwarg_name] = kwarg
            cur_args_copy = tuple(cur_args_copy)
        else:
            assert cur_node_orig.target not in BINARY_FUNCTIONS
            cur_args_copy = [cur_node_copy]
            if len(cur_node_orig.args) > 1:
                for arg in cur_node_orig.args[1:]:
                    if isinstance(arg, torch.nn.Parameter):
                        new_arg = arg.clone().detach()
                        mod_name = f'mod_{cur_name_idx}'
                        cur_name_idx += 1
                        setattr(gm, mod_name, new_arg)
                        new_arg_placeholder = gm.placeholder(mod_name)
                        cur_args_copy.append(new_arg_placeholder)
                    elif isinstance(arg, (float, int, torch.dtype)):
                        cur_args_copy.append(arg)
                    else:
                        raise AssertionError(f'arg of type {type(arg)} not handled yet')
            cur_args_copy = tuple(cur_args_copy)
        if cur_node_orig.op == 'call_module':
            orig_mod = getattr_from_fqn(model, cur_node_orig.target)
            orig_mod_copy = copy.deepcopy(orig_mod)
            mod_name = f'mod_{cur_name_idx}'
            setattr(gm, mod_name, orig_mod_copy)
            cur_name_idx += 1
            cur_node_copy = g.call_module(mod_name, cur_args_copy, cur_kwargs_copy)
        elif cur_node_orig.op == 'call_function':
            cur_node_copy = g.call_function(cur_node_orig.target, cur_args_copy, cur_kwargs_copy)
        elif cur_node_orig.op == 'call_method':
            cur_node_copy = g.call_method(cur_node_orig.target, cur_args_copy, cur_kwargs_copy)
        else:
            raise AssertionError(f'{cur_node_orig.op} not supported yet')
        if cur_node_orig is last_node:
            break
        assert len(cur_node_orig.users.keys()) == 1, f'{cur_node_orig} has more than 1 users, not supported yet'
        cur_node_orig = next(iter(cur_node_orig.users.keys()))
        cur_args_orig = cur_node_orig.args
        cur_kwargs_orig = cur_node_orig.kwargs
        cur_iteration += 1
        if cur_iteration > iteration_limit:
            raise AssertionError('iteration limit exceeded')
    g.output(cur_node_copy)
    gm.recompile()
    return gm