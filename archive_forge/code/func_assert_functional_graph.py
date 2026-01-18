import torch
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx.experimental.symbolic_shapes import definitely_true, sym_eq
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
from .schemas import MutationType
def assert_functional_graph(fx_g: torch.fx.Graph, *, allow_input_mutations: bool=False) -> int:
    placeholders = set()
    copy_count = 0
    for n in fx_g.nodes:
        if n.op == 'placeholder':
            placeholders.add(n)
        if isinstance(n.target, torch._ops.OpOverload):
            if n.target is torch.ops.aten.copy_.default and allow_input_mutations:
                suffix = True
                assert n.args[0] in placeholders
                placeholders.remove(n.args[0])
                copy_count += 1
            else:
                assert not n.target._schema.is_mutable, f'aot_autograd expected to have an entirely functional graph, but found {n.format_node()}'
    return copy_count