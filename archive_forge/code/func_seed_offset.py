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
def seed_offset(self, name, value):
    if value in self.sizevars:
        return self.sizevars[value]
    if name in self.sizevars.values():
        name = f'{name}{sum((1 for v in self.sizevars.values() if v.startswith(name)))}'
    self.sizevars[value] = name
    return name