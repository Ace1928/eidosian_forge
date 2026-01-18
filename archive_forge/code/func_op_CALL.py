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
def op_CALL(self, inst: dis.Instruction):
    argc: int = inst.argval
    arg1plus = reversed([self.pop() for _ in range(argc)])
    arg0 = self.pop()
    kw_names = self.pop_kw_names()
    args: list[ValueState] = [arg0, *arg1plus]
    callable = self.pop()
    opname = 'call' if kw_names is None else 'call.kw'
    op = Op(opname=opname, bc_inst=inst)
    op.add_input('env', self.effect)
    op.add_input('callee', callable)
    for i, arg in enumerate(args):
        op.add_input(f'arg.{i}', arg)
    if kw_names is not None:
        op.add_input('kw_names', kw_names)
    self.replace_effect(op.add_output('env', is_effect=True))
    self.push(op.add_output('ret'))