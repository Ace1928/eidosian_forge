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
def compute_buffer_groups(self, lines):
    """
        Populates self.buffer_groups with BufferGroup objects that join
        allocations with common storage (due to inplace reuse) into a
        single object.
        """
    name_to_group = {}
    for line in lines:
        if isinstance(line, AllocateLine):
            name = line.node.get_name()
            assert name not in name_to_group
            name_to_group[name] = BufferGroup(line.node)
        elif isinstance(line, ReuseLine):
            old_name = line.node.get_name()
            new_name = line.reused_as.get_name()
            assert new_name not in name_to_group
            if old_name in name_to_group:
                name_to_group[old_name].names.append(new_name)
                name_to_group[new_name] = name_to_group[old_name]
    outputs = set(V.graph.get_output_names())
    unique_groups = [*{id(g): g for g in name_to_group.values()}.values()]
    for group in unique_groups:
        group.is_output = any((x in outputs for x in group.names))
    assert self.buffer_groups is None
    self.buffer_groups = unique_groups
    return name_to_group