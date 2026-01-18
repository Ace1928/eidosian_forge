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
def convert_to_pool_lines(self, lines):
    """
        Convert AllocateLine/FreeIfNotReusedLine/ReuseLine into their
        pool-based counterparts.
        """
    name_to_group = self.compute_buffer_groups(lines)
    for i, line in enumerate(lines):
        if isinstance(line, AllocateLine):
            if line.node.get_name() in name_to_group:
                lines[i] = AllocFromPoolLine(self.wrapper, name_to_group[line.node.get_name()])
        elif isinstance(line, FreeIfNotReusedLine):
            assert not line.is_reused
            if line.node.get_name() in name_to_group:
                lines[i] = DeallocFromPoolLine(self.wrapper, name_to_group[line.node.get_name()])
        elif isinstance(line, ReuseLine):
            if line.node.get_name() in name_to_group:
                line.delete_old = False