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
def live_output_buffers(self):
    live_outs = set()
    for inplaced in unique(self.inplace_buffers.values()):
        if self._buffer_is_marked_removed(inplaced):
            continue
        live_outs.add(inplaced.other_names[-1])
    for outer, inner in self.output_buffers.items():
        if outer in self.inplace_buffers or self._buffer_is_marked_removed(inner):
            continue
        live_outs.add(outer)
    return live_outs