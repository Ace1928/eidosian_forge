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
def create_fx_call_function(self, op: Callable, args: Tuple) -> Tuple[Optional[torch.fx.Node], bool]:
    node_key = (op, args)
    fresh = False
    if self._translation_validation_enabled and node_key not in self.fx_node_cache:
        from torch.fx.experimental.validator import z3op
        if any((a is None for a in args)):
            assert all((not isinstance(a, torch.fx.Node) for a in args))
            return (None, fresh)
        fresh = True
        lifted_op = z3op(op, self.validator)
        assert all((a is not None for a in args)), f'missing arg in FX graph ({op.__name__}): {args}'
        node = self.fx_node_cache[node_key] = self.graph.call_function(lifted_op, args)
        self.name_to_node[node.name] = node
    return (self.fx_node_cache.get(node_key, None), fresh)