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
def canonicalize_bool_expr(expr: sympy.Expr):
    """ Canonicalize a boolean expression by transforming it into a lt / le
    inequality and moving all the non-constant terms to the rhs.
    We canonicalize And / Ors / Not via cnf and then canonicalize their subexpr
    recursively
    nb. sympy.Rel.canonical is not good enough https://github.com/sympy/sympy/issues/25924

    Args:
        expr (sympy.Expr): Expression to canonicalize
    """
    if not isinstance(expr, (sympy.Rel, sympy.And, sympy.Or, sympy.Not, sympy.Eq, sympy.Ne)):
        return expr
    if isinstance(expr, (sympy.And, sympy.Or, sympy.Not)):
        expr = sympy.logic.boolalg.to_cnf(expr)
    return _canonicalize_bool_expr_impl(expr)