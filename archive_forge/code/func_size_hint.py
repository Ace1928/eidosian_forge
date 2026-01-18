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
@lru_cache(256)
def size_hint(self, expr: 'sympy.Expr', *, allow_none=False):
    """
        Gets a size hint for a given expression from the underlying shapes we had.
        Does not introduce a guard, so only use this when you can guarantee that
        your code is still valid for arbitrary shapes (such as optimization decisions)
        """
    result_expr = safe_expand(expr).xreplace(self.var_to_val)
    if not result_expr.is_number:
        r = self._maybe_evaluate_static(result_expr, compute_hint=True)
        if r is not None:
            return r
        if allow_none:
            return None
        raise self._make_data_dependent_error(result_expr, expr)
    return result_expr