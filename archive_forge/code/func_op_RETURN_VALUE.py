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
def op_RETURN_VALUE(self, inst: dis.Instruction):
    tos = self.pop()
    op = Op(opname='ret', bc_inst=inst)
    op.add_input('env', self.effect)
    op.add_input('retval', tos)
    self.replace_effect(op.add_output('env', is_effect=True))