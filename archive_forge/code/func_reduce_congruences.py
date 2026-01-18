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
def reduce_congruences(self):
    reduced_congruences = {}
    for s, congruences in self._congruences.items():
        remainder_modulus_pairs = []
        congruences_to_check = set()
        for congruence in congruences:
            base, divisor = congruence.args
            tmp = sympy.Symbol('tmp', integer=True)
            symbol, solution = sympy.solve_linear(base - divisor * tmp, symbols=[s])
            if s == symbol:
                modulus, remainder = sympy.polys.polytools.div(solution, tmp)
                if isinstance(modulus, sympy.Integer) and isinstance(remainder, sympy.Integer):
                    remainder = remainder % modulus
                    remainder_modulus_pairs.append((remainder, modulus))
                    continue
            congruences_to_check.add(congruence)
        if remainder_modulus_pairs:
            remainder, modulus = sympy.ntheory.modular.solve_congruence(*remainder_modulus_pairs)
            reduced_congruences[s] = {(s - remainder) % modulus}
            substitution = {s: modulus * sympy.Symbol('tmp', integer=True) + remainder}
            reduced_congruences[s].update((congruence for congruence in congruences_to_check if not sympy.checksol(congruence, substitution)))
        else:
            reduced_congruences[s] = congruences_to_check
    return reduced_congruences