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
def compute_live_ranges(self, lines):
    """Populate every BufferGroup.live_ranges field based on first/last usage"""
    timestep = 0
    worklist = collections.deque(lines)
    while worklist:
        if isinstance(worklist[0], MemoryPlanningLine):
            timestep += 1
            while worklist and isinstance(worklist[0], MemoryPlanningLine):
                line = worklist.popleft()
                if isinstance(line, PoolMemoryPlanningLine):
                    line.group.update_usage(timestep)
                    line.timestep = timestep
        else:
            worklist.popleft()
    timestep += 1
    assert self.buffer_groups is not None
    for group in self.buffer_groups:
        if group.is_output:
            group.update_usage(timestep)