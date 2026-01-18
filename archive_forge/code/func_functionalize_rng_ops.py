from torch.fx.experimental.proxy_tensor import is_sym_node, py_sym_types
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch.fx.experimental.symbolic_shapes import (
import torch
import torch.fx as fx
import operator
import math
import torch.utils._pytree as pytree
import copy
import os
import itertools
import sympy
from collections import defaultdict
from torch.fx.passes import graph_drawer
from typing import List, Optional, Tuple, Union
from .compile_utils import fx_graph_cse, get_aten_target
from . import config
import functools
def functionalize_rng_ops(joint_module, fw_module, bw_module, num_sym_nodes):
    uid = itertools.count()

    def get_rng_ops(gmod):
        random_nodes = {}
        for node in gmod.graph.nodes:
            if node.op == 'call_function' and hasattr(node.target, 'tags') and (torch.Tag.nondeterministic_seeded in node.target.tags):
                random_nodes[node.name] = node
        return random_nodes

    def get_device(node):
        """
        Check the example value of the node outputs to find the device type.
        """
        if 'val' not in node.meta:
            return None
        candidates = node.meta['val']
        if not isinstance(candidates, tuple):
            candidates = (candidates,)
        for candidate in candidates:
            if isinstance(candidate, torch.Tensor):
                if candidate.device.type == 'cuda':
                    return 'cuda'
        return 'cpu'

    def get_sample_rng_state(device):
        if device == 'cuda':
            return torch.cuda.get_rng_state()
        return torch.get_rng_state()
    joint_graph_rng_ops = get_rng_ops(joint_module)
    fw_graph_rng_ops = get_rng_ops(fw_module)
    bw_graph_rng_ops = get_rng_ops(bw_module)
    recomputable_rng_ops_map = dict()
    for node in joint_module.graph.nodes:
        if must_recompute(node) and hasattr(node.target, 'tags') and (torch.Tag.nondeterministic_seeded in node.target.tags):
            base_node = joint_graph_rng_ops[node.name]
            fw_node = fw_graph_rng_ops[node.name]
            bw_node = bw_graph_rng_ops[node.name]
            recomputable_rng_ops_map[base_node] = {'fwd': fw_node, 'bwd': bw_node}
    run_and_save_rng = torch._prims.rng_prims.run_and_save_rng_state
    run_with_rng_state = torch._prims.rng_prims.run_with_rng_state
    for node in bw_module.graph.nodes:
        if node.op == 'placeholder' and 'tangent' in node.name:
            bw_tangent_start_node = node
            break
    fw_rng_state_outputs = []
    for base_node, node_pair in recomputable_rng_ops_map.items():
        fw_node = node_pair['fwd']
        bw_node = node_pair['bwd']
        fw_graph = fw_module.graph
        with fw_graph.inserting_before(fw_node):
            functional_fw_node = fw_graph.create_node('call_function', run_and_save_rng, args=(fw_node.target, *fw_node.args), kwargs=fw_node.kwargs)
            state = fw_graph.create_node('call_function', operator.getitem, args=(functional_fw_node, 0), kwargs={})
            rng_output = fw_graph.create_node('call_function', operator.getitem, args=(functional_fw_node, 1), kwargs={})
            fw_node.replace_all_uses_with(rng_output)
            fw_graph.erase_node(fw_node)
            fw_rng_state_outputs.append(state)
        bw_graph = bw_module.graph
        with bw_graph.inserting_before(bw_tangent_start_node):
            state_name = f'rng_state_output_{next(uid)}'
            bw_rng_state_node = bw_graph.placeholder(state_name)
            bw_rng_state_node.meta['val'] = get_sample_rng_state(get_device(fw_node))
        with bw_graph.inserting_before(bw_node):
            rng_output = bw_graph.create_node('call_function', run_with_rng_state, args=(bw_rng_state_node, bw_node.target, *bw_node.args), kwargs=bw_node.kwargs)
            bw_node.replace_all_uses_with(rng_output)
            bw_graph.erase_node(bw_node)
    fw_output_node = next((node for node in fw_module.graph.nodes if node.op == 'output'))
    fw_outputs = fw_output_node.args[0]
    sym_node_start_idx = len(fw_outputs) - num_sym_nodes
    outputs = fw_outputs[:sym_node_start_idx] + fw_rng_state_outputs + fw_outputs[sym_node_start_idx:]
    fw_module.graph.output(outputs)
    fw_module.graph.erase_node(fw_output_node)
    fw_module.recompile()
    bw_module.recompile()
    return (fw_module, bw_module)