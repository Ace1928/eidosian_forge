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
def convert_bc_to_ddg(block: PythonBytecodeBlock, bcmap: dict[int, dis.Bytecode], argnames: tuple[str, ...]) -> DDGBlock:
    instlist = block.get_instructions(bcmap)
    return _convert_bytecode(block, instlist, argnames)