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
def indirect_load(self, name: str, index: sympy.Expr):
    """A load the depends on an index we have read"""
    prior = self.loads
    try:
        self.loads = self.compute
        return self.load(name, index)
    finally:
        self.loads = prior