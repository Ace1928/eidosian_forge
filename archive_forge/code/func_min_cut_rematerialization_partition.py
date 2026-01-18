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
def min_cut_rematerialization_partition(joint_module: fx.GraphModule, _joint_inputs, compiler='inductor', recomputable_ops=None, *, num_fwd_outputs) -> Tuple[fx.GraphModule, fx.GraphModule]:
    """
    Partitions the joint graph such that the backward recomputes the forward.
    Recomputing helps in trading off memory bandwidth with computation.

    To create the fwd and bwd graph, we copy the joint graph, manually set the
    outputs to just original forward or backward outputs. And then we run the
    resulting graphs through dead code elimination.

    .. warning::
        This API is experimental and likely to change.

    Args:
        joint_module(fx.GraphModule): The joint forward and backward graph. This
            is the result of AOT Autograd tracing.
        _joint_inputs: The inputs to the joint graph. This is unused.
        compiler: This option determines the default set of recomputable ops.
            Currently, there are two options: ``nvfuser`` and ``inductor``.
        recomputable_ops: This is an optional set of recomputable ops. If this
            is not None, then this set of ops will be used instead of the
            default set of ops.
        num_fwd_outputs: The number of outputs from the forward graph.

    Returns:
        Returns the generated forward and backward Fx graph modules.
    """
    try:
        import networkx as nx
    except ImportError as e:
        raise RuntimeError('Need networkx installed to perform smart recomputation heuristics') from e
    joint_module.graph.eliminate_dead_code()
    joint_module.recompile()
    fx_g = joint_module.graph
    if config.cse:
        cse_graph = fx_graph_cse(fx_g)
        joint_module.graph = cse_graph
    full_bw_graph = joint_module.graph
    graph_has_recomputable_ops = has_recomputable_ops(joint_module)
    graph_has_recomputable_rng_ops = has_recomputable_rng_ops(joint_module)
    if graph_has_recomputable_ops:
        joint_module = cleanup_recompute_tags(joint_module)
    name_to_node = {}
    for node in joint_module.graph.nodes:
        name_to_node[node.name] = node

    def classify_nodes(joint_module):
        required_bw_nodes = set()
        for node in joint_module.graph.nodes:
            if node.op == 'placeholder' and 'tangents' in node.target:
                required_bw_nodes.add(node)
            if node in required_bw_nodes:
                for user in node.users:
                    required_bw_nodes.add(user)
        primal_inputs = list(filter(_is_primal, joint_module.graph.nodes))
        fwd_seed_offset_inputs = list(filter(_is_fwd_seed_offset, joint_module.graph.nodes))
        inputs = primal_inputs + fwd_seed_offset_inputs
        fwd_outputs, bwd_outputs = _extract_fwd_bwd_outputs(joint_module, num_fwd_outputs=num_fwd_outputs)
        required_bw_nodes.update((o for o in bwd_outputs if o is not None))
        forward_only_graph = _extract_graph_with_inputs_outputs(joint_module.graph, inputs, fwd_outputs)
        required_fw_nodes = {name_to_node[node.name] for node in forward_only_graph.nodes if node.op != 'output'}
        unclaimed_nodes = {node for node in joint_module.graph.nodes if node not in required_fw_nodes and node not in required_bw_nodes}
        return (fwd_outputs, required_fw_nodes, required_bw_nodes, unclaimed_nodes)
    orig_fw_outputs, required_fw_nodes, required_bw_nodes, unclaimed_nodes = classify_nodes(joint_module)
    if len(required_bw_nodes) == 0:
        return default_partition(joint_module, _joint_inputs, num_fwd_outputs=num_fwd_outputs)
    for node in reversed(joint_module.graph.nodes):
        if node not in required_fw_nodes:
            node.dist_from_bw = 0
        else:
            node.dist_from_bw = int(1000000000.0)
            for user in node.users:
                node.dist_from_bw = min(node.dist_from_bw, user.dist_from_bw + 1)
    aten = torch.ops.aten
    prims = torch.ops.prims
    default_recomputable_ops = [aten.add, aten.sub, aten.div, aten.atan2, aten.mul, aten.max, aten.min, aten.pow, aten.remainder, aten.fmod, aten.__and__, aten.__or__, aten.__xor__, aten.__lshift__, aten.__rshift__, aten.eq, aten.ne, aten.ge, aten.gt, aten.le, aten.lt, aten.abs, aten.bitwise_not, aten.ceil, aten.floor, aten.frac, aten.neg, aten.relu, aten.round, aten.silu, aten.trunc, aten.log, aten.log10, aten.log1p, aten.log2, aten.lgamma, aten.exp, aten.expm1, aten.erf, aten.erfc, aten.cos, aten.acos, aten.cosh, aten.sin, aten.asin, aten.sinh, aten.tan, aten.atan, aten.tanh, aten.atanh, aten.sqrt, aten.rsqrt, aten.reciprocal, aten.sigmoid, aten.softplus, aten.threshold, aten.threshold_backward, aten.clamp, aten.where, aten.lerp, aten.addcmul, aten.gelu, aten.gelu_backward, aten.sum, aten.mean, aten._grad_sum_to_size, aten.sum_to_size, aten.amax, aten.to, aten.type_as, operator.getitem, aten.squeeze, aten.unsqueeze, aten.rsub, aten._to_copy]
    view_ops = [aten.squeeze, aten.unsqueeze, aten.alias]
    if compiler == 'inductor':
        default_recomputable_ops += [prims.div, prims.convert_element_type, aten.clone, aten._to_copy, aten.full_like, prims.var, prims.sum, aten.var, aten.std, prims.broadcast_in_dim, aten.select, aten.permute, aten._unsafe_view, aten.view, aten.expand, aten.slice, aten.reshape, aten.broadcast_tensors, aten.scalar_tensor, aten.ones, aten.new_zeros, aten.lift_fresh_copy, aten.arange, aten.triu, aten.var_mean, aten.isinf, aten.any, aten.full, aten.as_strided, aten.zeros, aten.argmax, aten.maximum]
        view_ops += [aten.view, aten.slice, aten.permute, aten.t, prims.broadcast_in_dim, aten.expand, aten.as_strided]
        default_recomputable_ops += [aten.index]
    default_recomputable_ops += view_ops
    default_recomputable_ops += pointwise_ops()
    default_recomputable_ops += [aten.zeros_like]
    default_recomputable_ops += [method_to_operator(m) for m in magic_methods]
    recomputable_ops = set(recomputable_ops) if recomputable_ops is not None else set(default_recomputable_ops)
    random_ops = [aten.native_dropout, aten.rand_like, aten.randn_like]
    compute_intensive_ops = [aten.mm, aten.convolution, aten.convolution_backward, aten.bmm, aten.addmm, aten.upsample_bilinear2d, aten._softmax, aten._softmax_backward_data, aten.native_layer_norm, aten.native_layer_norm_backward, aten.native_batch_norm, aten.native_batch_norm_backward, aten._native_batch_norm_legit]
    unrecomputable_ops = random_ops + compute_intensive_ops
    fusible_ops = recomputable_ops | set(random_ops)
    if AOT_PARTITIONER_DEBUG:
        joint_module_ops = {str(node.target._overloadpacket) for node in joint_module.graph.nodes if node.op == 'call_function' and hasattr(node.target, '_overloadpacket')}
        ops_ignored = joint_module_ops - {str(i) for i in recomputable_ops}
        print('Ops banned from rematerialization: ', ops_ignored)
        print()
    AGGRESSIVE_RECOMPUTATION = False

    def is_materialized_backwards(node):
        cur_nodes = {node}
        while len(cur_nodes) > 0:
            cur = cur_nodes.pop()
            for user in cur.users:
                if user not in required_fw_nodes and (not is_fusible(cur, user)):
                    return True
                if user not in required_fw_nodes and get_aten_target(user) in view_ops:
                    cur_nodes.add(user)
        return False

    def ban_recomputation(node):
        if 'recompute' in node.meta:
            return node.meta['recompute'] == 0
        elif AGGRESSIVE_RECOMPUTATION:
            return node.op == 'call_function' and get_aten_target(node) in unrecomputable_ops
        else:
            if node.op != 'call_function':
                return False
            if get_aten_target(node) not in recomputable_ops:
                return True
            if node.target == operator.getitem:
                return False
            if node.target in [aten.lift_fresh_copy.default, aten.lift_fresh.default]:
                return False
            if is_materialized_backwards(node):
                return True
            if not graph_has_recomputable_ops:
                if compiler == 'inductor' and node.dist_from_bw > config.max_dist_from_bw:
                    return True
            input_tensors_size = sum((_size_of(i) for i in node.args if isinstance(i, fx.Node)))
            output_size = _size_of(node)
            return output_size * 4 < input_tensors_size

    def is_fusible(a, b):
        if get_aten_target(b) == aten.cat:
            return True
        return get_aten_target(a) in fusible_ops and get_aten_target(b) in fusible_ops

    def is_materialized(node):
        if node.op == 'placeholder':
            return True
        return not all((is_fusible(node, user) for user in node.users))

    def get_node_weight(node) -> int:
        mem_sz = _size_of(node)
        mem_sz = int(mem_sz * 1.1 ** max(min(node.dist_from_bw, 100), 1))
        if is_materialized(node):
            return mem_sz
        else:
            return mem_sz * 2
    nx_graph = nx.DiGraph()
    for node in full_bw_graph.nodes:
        if node.op == 'output':
            continue
        if node in required_bw_nodes:
            nx_graph.add_edge(node.name + '_in', 'sink', capacity=math.inf)
            continue
        if _is_primal(node) or _is_fwd_seed_offset(node):
            nx_graph.add_edge('source', node.name + '_in', capacity=math.inf)
        if ban_recomputation(node) and node in required_fw_nodes:
            nx_graph.add_edge('source', node.name + '_in', capacity=math.inf)
        is_non_tensor_node = 'val' not in node.meta and 'tensor_meta' not in node.meta or ('val' in node.meta and (not isinstance(node.meta['val'], torch.Tensor)))
        if is_sym_node(node):
            weight = sym_node_size(node)
        elif is_non_tensor_node:
            weight = math.inf
        else:
            weight = get_node_weight(node)
        nx_graph.add_edge(node.name + '_in', node.name + '_out', capacity=weight)
        for user in node.users:
            nx_graph.add_edge(node.name + '_out', user.name + '_in', capacity=math.inf)
    try:
        cut_value, partition = nx.minimum_cut(nx_graph, 'source', 'sink')
    except Exception:
        print('Failed to compute min-cut on following graph:')
        print('\n'.join(nx.readwrite.edgelist.generate_edgelist(nx_graph)))
        raise
    reachable, non_reachable = partition
    cutset = set()
    for u, nbrs in ((n, nx_graph[n]) for n in reachable):
        cutset.update(((u, v) for v in nbrs if v in non_reachable))
    cut_nodes = set()
    for node_in, node_out in cutset:
        assert node_in[:-3] == node_out[:-4]
        node_name = node_in[:-3]
        cut_nodes.add(node_name)
    node_idx = {node: idx for idx, node in enumerate(joint_module.graph.nodes)}
    saved_values = sorted((name_to_node[node] for node in cut_nodes), key=lambda x: node_idx[x])
    saved_sym_nodes = list(filter(is_sym_node, saved_values))
    saved_values = list(filter(lambda n: not is_sym_node(n), saved_values))
    fw_module, bw_module = _extract_fwd_bwd_modules(joint_module, saved_values, saved_sym_nodes=saved_sym_nodes, num_fwd_outputs=num_fwd_outputs)
    if graph_has_recomputable_ops:
        if graph_has_recomputable_rng_ops:
            fw_module, bw_module = functionalize_rng_ops(joint_module, fw_module, bw_module, len(saved_sym_nodes))
        bw_module = reordering_to_mimic_autograd_engine(bw_module)
    if AOT_PARTITIONER_DEBUG:
        print('Theoretical Activations Stored: ', sum([_size_of(i) for i in saved_values]) / 1000000000.0)
        fw_module_nodes = {node.name for node in fw_module.graph.nodes if node.op == 'call_function'}
        bw_module_nodes = {node.name for node in bw_module.graph.nodes if node.op == 'call_function'}
        remat_nodes = fw_module_nodes & bw_module_nodes
        counts = defaultdict(int)
        for node in fw_module.graph.nodes:
            if node.name in remat_nodes and hasattr(node.target, '_overloadpacket'):
                counts[str(node.target._overloadpacket)] += 1
        print(f'# remat/fw/bw: {len(remat_nodes)}/{len(fw_module_nodes)}/{len(bw_module_nodes)}')
        print('Count of Ops Rematerialized: ', sorted(counts.items(), key=lambda x: x[1], reverse=True))
    return (fw_module, bw_module)