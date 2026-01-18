import os
from dataclasses import dataclass, replace, field, fields
import dis
import operator
from functools import reduce
from typing import (
from collections import ChainMap
from numba_rvsdg.core.datastructures.byte_flow import ByteFlow
from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.basic_block import (
from numba_rvsdg.rendering.rendering import ByteFlowRenderer
from numba_rvsdg.core.datastructures import block_names
from numba.core.utils import MutableSortedSet, MutableSortedMap
from .regionpasses import (
def get_toposorted_ops(self) -> list[Op]:
    """Get a topologically sorted list of ``Op`` according
        to the data-dependence.

        Operations stored later in the list may depend on earlier operations,
        but the reverse can never be true.
        """
    res: list[Op] = []
    avail: set[ValueState] = {*self.in_vars.values(), _just(self.in_effect)}
    pending: list[Op] = [vs.parent for vs in self.out_vars.values() if vs.parent is not None]
    assert self.out_effect is not None
    pending.append(_just(self.out_effect.parent))
    seen: set[Op] = set()
    while pending:
        op = pending[-1]
        if op in seen:
            pending.pop()
            continue
        incomings = set()
        for vs in op._inputs.values():
            if vs not in avail and vs.parent is not None:
                incomings.add(vs.parent)
        if not incomings:
            avail |= set(op._outputs.values())
            pending.pop()
            res.append(op)
            seen.add(op)
        else:
            pending.extend(incomings)
    return res