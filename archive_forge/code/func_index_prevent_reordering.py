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
def index_prevent_reordering(index: List[sympy.Expr], index_vars, sizes):
    from ..ir import FlexibleLayout
    return [*index, sympy_dot(index_vars, FlexibleLayout.contiguous_strides(sizes))]