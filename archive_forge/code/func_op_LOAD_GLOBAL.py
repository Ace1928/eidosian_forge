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
def op_LOAD_GLOBAL(self, inst: dis.Instruction):
    assert isinstance(inst.arg, int)
    load_null = inst.arg & 1
    op = Op(opname='global', bc_inst=inst)
    op.add_input('env', self.effect)
    null = op.add_output('null')
    if load_null:
        self.push(null)
    self.push(op.add_output(f'{inst.argval}'))