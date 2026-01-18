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
def set_current_node(self, node):
    prior = self.current_node
    self.current_node = node
    self.node_to_bounds = node._body.bounds().get_bounds()
    try:
        yield
    finally:
        self.current_node = prior