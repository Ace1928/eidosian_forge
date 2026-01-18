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
@contextlib.contextmanager
def ctx():
    for _ in range(offset):
        self.writeline('{')
        self._indent += 1
    for _ in range(-offset):
        self._indent -= 1
        self.writeline('}')
    yield
    for _ in range(-offset):
        self.writeline('{')
        self._indent += 1
    for _ in range(offset):
        self._indent -= 1
        self.writeline('}')