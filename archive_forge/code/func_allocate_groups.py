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
def allocate_groups(self):
    """
        Assign every allocation to a specific location in a specific AllocationPool.
        """
    assert config.memory_pool in ('none', 'intermediates', 'outputs', 'combined')
    assert self.buffer_groups is not None
    for group in self.buffer_groups:
        group.make_allocation()
    outputs: List[Allocation] = []
    intermediates: List[Allocation] = []
    for group in self.buffer_groups:
        assert group.allocation
        if group.is_output and config.memory_pool != 'combined':
            outputs.append(group.allocation)
        else:
            intermediates.append(group.allocation)
    for block in sorted(outputs, key=lambda x: (x.size_hint, -len(x.live_range))):
        self.pools.allocate_output(block)
    for block in sorted(intermediates, key=lambda x: (-x.size_hint, -len(x.live_range))):
        self.pools.allocate(block)
    self.pools.finalize()