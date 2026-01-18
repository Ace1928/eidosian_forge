import collections
import dataclasses
import itertools
import logging
import re
import typing
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import sympy
import torch
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from .codegen.common import index_prevent_reordering
from .utils import get_dtype_size, sympy_str, sympy_subs, sympy_symbol, VarRanges
from .virtualized import V
def extract_input_node_reduction_ranges(input_node: 'torch._inductor.ir.TensorBox') -> Tuple[Optional[List[sympy.Expr]], Optional[List[sympy.Expr]]]:
    """
    Returns the size and reduction size of all inputs, if the sizes and reduction_sizes (if exist) are all the same.
    It's possible that a node has multiple inputs, some are Reduction nodes and others are Pointwise nodes.
    In this case, reduction_sizes of the Reduction nodes need to be the same.
    Otherwise returns (None, None).
    """
    from .ir import ComputedBuffer, Loops
    if isinstance(input_node.data, ComputedBuffer):
        size = input_node.get_size()
        reduction_size = input_node.get_reduction_size()
        if len(reduction_size) > 0:
            return (size, reduction_size)
        else:
            return (None, None)
    if not isinstance(input_node.data.data, Loops):
        return (None, None)
    reads = input_node.get_reads()
    reduction_size = None
    size = None
    while reduction_size is None and len(reads) > 0:
        seen = set()
        new_reads = []
        for read in reads:
            if not isinstance(read, MemoryDep):
                continue
            if read.name in seen:
                continue
            seen.add(read.name)
            buffer = V.graph.get_buffer(read.name)
            if buffer is None:
                continue
            if isinstance(buffer, ComputedBuffer) and len(buffer.get_reduction_size()) > 0:
                if reduction_size is None:
                    reduction_size = buffer.get_reduction_size()
                    size = buffer.get_size()
                elif reduction_size != buffer.get_reduction_size() or size != buffer.get_size():
                    return (None, None)
            else:
                new_reads.extend(buffer.get_reads())
        if reads == new_reads:
            return (size, reduction_size)
        else:
            reads = new_reads
    return (size, reduction_size)