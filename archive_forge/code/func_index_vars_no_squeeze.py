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
def index_vars_no_squeeze(*argsizes: Tuple[sympy.Expr, ...], prefix: str):
    var_ranges, add_var = var_builder(prefix)
    args: List[List[sympy.Symbol]] = []
    for size in argsizes:
        args.append(list(map(add_var, size)))
    return (args, var_ranges)