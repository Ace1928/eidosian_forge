from __future__ import annotations
import collections
import dataclasses
import itertools
import pprint
from typing import Any, Dict, Iterable, List, Optional, Protocol
import sympy
import torch
from .. import config, ir
from ..utils import cache_on_self, CachedMethod, IndentedBuffer
from ..virtualized import V
from .wrapper import (
def allocate_at_end(self, block):
    block.mark_allocated()
    self.root = TemporalSplit([SpatialSplit(self.root, TemporalSplit([block]))])
    return True