import dis
from contextlib import contextmanager
import builtins
import operator
from typing import Iterator
from functools import reduce
from numba.core import (
from numba.core.utils import (
from .rvsdg.bc2rvsdg import (
from .rvsdg.regionpasses import RegionVisitor
@property
def last_block(self) -> ir.Block:
    assert self.last_block_label is not None
    return self.blocks[self.last_block_label]