from __future__ import annotations
import dataclasses
import functools
import inspect
import itertools
import logging
import os
import re
from collections import defaultdict
from typing import (
from typing_extensions import TypeGuard
import torch
import torch._guards
import torch.fx
import torch.utils._pytree as pytree
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import counters
from torch._prims_common import is_integer_dtype
from torch.fx import Node
from torch.fx.experimental.proxy_tensor import make_fx, maybe_disable_fake_tensor_mode
from torch.fx.immutable_collections import immutable_dict, immutable_list
from .._functorch import config as functorch_config
from .._functorch.aot_autograd import aot_function, make_boxed_func
from .._functorch.partitioners import default_partition
from .._subclasses import FakeTensorMode
from ..fx import Transformer
from . import config
from .decomposition import select_decomp_table
from .lowering import fallback_node_due_to_unsupported_type
@staticmethod
def replace_with_graph(match: Match, graph: torch.fx.Graph, replacement_graph: torch.fx.Graph, args: List[Any]):
    output_nodes = match.output_nodes()
    first_node = output_nodes[0]

    class Replacer(torch.fx.Interpreter):
        call_method = None
        call_module = None
        get_attr = None

        def run_node(self, node) -> Any:
            if node.op in ('placeholder', 'output'):
                return super().run_node(node)
            if node.op == 'call_function':
                target = node.target
                args, kwargs = self.fetch_args_kwargs_from_env(node)
                result = graph.call_function(target, args, kwargs)
                if 'val' in node.meta and 'val' not in result.meta:
                    result.meta['val'] = node.meta['val']
                    if isinstance(node.meta['val'], torch.Tensor):
                        assert 'tensor_meta' in node.meta
                        result.meta['tensor_meta'] = node.meta['tensor_meta']
                return result
            raise NotImplementedError(f'unhandled {node}')
    output_nodes = match.output_nodes()
    if len(output_nodes) == 1:
        last_node = output_nodes[0]
    else:
        assert output_nodes[0]
        nodes = list(output_nodes[0].graph.nodes)
        indices = [(nodes.index(n), n) for n in output_nodes if isinstance(n, torch.fx.Node)]
        last_node = min(indices, key=lambda tup: tup[0])[1]

    def percolate_tags(node, recompute_tag):
        for arg in node.all_input_nodes:
            if hasattr(arg, 'meta'):
                arg.meta['recompute'] = recompute_tag
                percolate_tags(arg, recompute_tag)
    with graph.inserting_before(last_node):
        replacement = Replacer(replacement_graph).run(*args)
        if isinstance(replacement, torch.fx.Node):
            replacement = [replacement]
        assert len(replacement) == len(output_nodes)
        for old, new in zip(output_nodes, replacement):
            if old is None:
                assert new is None
            elif new is None:
                old.replace_all_uses_with(None)
            else:
                if 'val' not in new.meta:
                    new.meta.update(old.meta)
                if 'recompute' in old.meta:
                    percolate_tags(new, old.meta['recompute'])
                old.replace_all_uses_with(new)
    match.erase_nodes(graph)