from __future__ import annotations
import collections
import dataclasses
import itertools
import pprint
from typing import Any, Dict, Iterable, List, Optional, Protocol
import sympy
import torch
from .. import config, ir
from ..utils import cache_on_self, CachedMethod, IndentedBuffer
from ..virtualized import V
from .wrapper import (
def codegen_create(self, wrapper, code: IndentedBuffer):
    assert self.name
    nbytes = self.root.get_symbolic_size()
    for block in self.root.allocations:
        if isinstance(block, Allocation) and nbytes == block.get_symbolic_size():
            node = block.node
            code.writeline(wrapper.make_allocation(self.name, device=self.device, dtype=node.get_dtype(), shape=tuple(node.get_size()), stride=tuple(node.get_stride())))
            self.creation_cache[block.codegen_alloc_from_pool(wrapper)] = self.name
            return
    else:
        code.writeline(wrapper.make_allocation(self.name, device=self.device, dtype=torch.uint8, shape=(nbytes,), stride=(1,)))