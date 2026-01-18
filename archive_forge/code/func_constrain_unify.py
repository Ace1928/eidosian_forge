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
def constrain_unify(a, b):
    """
    Given two SymInts, constrain them so that they must be equal.  NB:
    this will not work with SymInts that represent nontrivial expressions
    (yet!)
    """
    if not isinstance(a, SymInt):
        if not isinstance(b, SymInt):
            assert a == b
        else:
            assert isinstance(b.node.expr, sympy.Symbol), 'constraining non-Symbols NYI'
            shape_env = b.node.shape_env
            shape_env.replacements[b.node.expr] = sympy.Integer(a)
    else:
        assert isinstance(a.node.expr, sympy.Symbol), 'constraining non-Symbols NYI'
        shape_env = a.node.shape_env
        if not isinstance(b, SymInt):
            shape_env.replacements[a.node.expr] = sympy.Integer(b)
        else:
            assert a.node.shape_env is b.node.shape_env
            assert isinstance(b.node.expr, sympy.Symbol), 'constraining non-Symbols NYI'
            new_var = shape_env._find(a.node.expr)
            shape_env.replacements[b.node.expr] = new_var