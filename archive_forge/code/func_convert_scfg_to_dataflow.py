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
def convert_scfg_to_dataflow(scfg, bcmap, argnames: tuple[str, ...]) -> SCFG:
    rvsdg = SCFG()
    for block in scfg.graph.values():
        if isinstance(block, PythonBytecodeBlock):
            ddg = convert_bc_to_ddg(block, bcmap, argnames)
            rvsdg.add_block(ddg)
        elif isinstance(block, RegionBlock):
            subregion = convert_scfg_to_dataflow(block.subregion, bcmap, argnames)
            rvsdg.add_block(_upgrade_dataclass(block, DDGRegion, dict(subregion=subregion)))
        elif isinstance(block, SyntheticBranch):
            rvsdg.add_block(_upgrade_dataclass(block, DDGBranch))
        elif isinstance(block, SyntheticAssignment):
            rvsdg.add_block(_upgrade_dataclass(block, DDGControlVariable))
        elif isinstance(block, ExtraBasicBlock):
            ddg = convert_extra_bb(block)
            rvsdg.add_block(ddg)
        elif isinstance(block, BasicBlock):
            start_env = Op('start', bc_inst=None)
            effect = start_env.add_output('env', is_effect=True)
            newblk = _upgrade_dataclass(block, DDGBlock, dict(in_effect=effect, out_effect=effect))
            rvsdg.add_block(newblk)
        else:
            raise Exception('unreachable', type(block))
    return rvsdg