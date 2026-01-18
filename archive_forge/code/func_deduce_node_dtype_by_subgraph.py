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
def deduce_node_dtype_by_subgraph(self, node: torch.fx.Node):
    sub_graph = self.graphs[node.target]
    dtype = self.propagate_graph(sub_graph)
    assert dtype
    return dtype