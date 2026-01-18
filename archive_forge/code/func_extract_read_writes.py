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
def extract_read_writes(fn: Callable[..., Any], *argsizes: Tuple[sympy.Expr, ...], normalize: bool=False, prefix: str='d'):
    args, var_ranges = index_vars_squeeze(*argsizes, prefix=prefix)
    rw = RecordLoadStore(var_ranges, normalize=normalize)
    with V.set_ops_handler(rw):
        fn(*args)
    if normalize:
        range_vars = []
    else:
        range_vars = [*itertools.chain(*args)]
    inner = rw.parent_handler.parent_handler
    return ReadWrites(set(inner._reads), set(inner._writes), inner._index_exprs, range_vars, var_ranges, rw.parent_handler._op_counts)