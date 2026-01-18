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
def drop_removed_buffers(self, lines):
    """
        Replace any memory planning lines in V.graph.removed_buffers with NullLine
        """
    for i, line in enumerate(lines):
        if isinstance(line, (AllocateLine, FreeIfNotReusedLine, ReuseLine)):
            if line.node.get_name() in V.graph.removed_buffers:
                lines[i] = NullLine(self.wrapper)