import builtins
import collections
import functools
import inspect
import itertools
import logging
import math
import operator
import re
import sys
import threading
import traceback
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, cast, Callable, Dict, List, Optional, Sequence, Set, Tuple, Type, Union, Iterable
import torch
import torch.fx
import torch.fx.traceback as fx_traceback
from torch.fx.experimental import _config as config
from torch.fx.experimental.recording import (
from torch.fx.experimental.sym_node import SymNode, SymTypes
from torch import SymBool, SymFloat, SymInt
from torch._guards import ShapeGuard, Source, TracingContext
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._sympy.functions import FloorDiv, Mod, IsNonOverlappingAndDenseIndicator
from torch.utils._sympy.solve import try_solve
from torch.utils._sympy.value_ranges import bound_sympy, SymPyValueRangeAnalysis, ValueRanges, ValueRangeError
from torch.utils._sympy.singleton_int import SingletonInt
from torch.utils._traceback import format_frame, CapturedTraceback
from torch._utils_internal import signpost_event
from torch._logging import LazyString
import sympy
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence, PRECEDENCE
@record_shapeenv_event()
def create_symintnode(self, sym: 'sympy.Expr', *, hint: Optional[int], source: Optional[Source]=None, symbolic_context: Optional[SymbolicContext]=None):
    source_name = source.name() if source else None
    if self._translation_validation_enabled and source is not None:
        symbol = self._create_symbol_for_source(source)
        assert symbol is not None
        fx_node = self.create_fx_placeholder_and_z3var(symbol, int)
        self._add_assertion(sympy.Eq(symbol, sym))
    else:
        fx_node = None
    if isinstance(symbolic_context, StatefulSymbolicContext) and source_name:
        if source_name in symbolic_context.source_to_symint_node_cache:
            return symbolic_context.source_to_symint_node_cache[source_name]
    if isinstance(sym, sympy.Integer):
        if hint is not None:
            assert int(sym) == hint
        out = int(sym)
    else:
        out = SymInt(SymNode(sym, self, int, hint, fx_node=fx_node))
    if isinstance(symbolic_context, StatefulSymbolicContext) and source_name:
        symbolic_context.source_to_symint_node_cache[source_name] = out
    return out