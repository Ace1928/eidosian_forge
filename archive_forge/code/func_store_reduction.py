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
@staticmethod
def store_reduction(name, index, value):
    self.store_buffer_names.add(name)
    self.cse.store_cache[name] = value
    if self.current_node:
        for other_name in self.current_node.get_mutations():
            self.cse.store_cache[other_name] = value
    if name not in V.graph.removed_buffers:
        return self.store_reduction(name, index, value)