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
def convert_extra_bb(block: ExtraBasicBlock) -> DDGBlock:
    converter = BC2DDG()
    for opname in block.inst_list:
        if opname == 'FOR_ITER_STORE_INDVAR':
            converter.push(converter.load('indvar'))
        elif opname == 'POP':
            converter.pop()
        else:
            assert False, opname
    return _converter_to_ddgblock(block, converter)