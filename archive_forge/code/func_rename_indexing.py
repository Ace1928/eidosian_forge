import contextlib
import dataclasses
import functools
import itertools
import logging
import operator
import re
from collections import namedtuple
from itertools import chain
from typing import (
import sympy
from sympy.printing.printer import Printer
import torch
import torch.fx
from torch.utils._sympy.value_ranges import ValueRanges
from .. import config, metrics
from ..utils import (
from ..virtualized import ops, OpsValue, V
def rename_indexing(self, index) -> sympy.Expr:
    if isinstance(index, (list, tuple)):
        return [self.rename_indexing(x) for x in index]
    index = V.graph.sizevars.simplify(index)
    sorted_symbols = sorted(index.free_symbols, key=lambda s: s.name)
    replacements = {x: self.args.size(x) for x in sorted_symbols if x.name.startswith('s') or x.name.startswith('ps') or (x.name.startswith('i') and (not x.name.startswith('idx')))}
    return sympy_subs(index, replacements)