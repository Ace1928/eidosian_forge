import copy
import dataclasses
import sys
import types
from typing import Any, cast, Dict, List, Optional, Tuple
from .bytecode_transformation import (
from .utils import ExactWeakKeyDictionary
def remap_block_offsets(instructions: List[Instruction], code_options: Dict[str, Any]):
    prefix_blocks: List[Instruction] = []
    for inst in instructions:
        if len(prefix_blocks) == len(meta.prefix_block_target_offset_remap):
            break
        if inst.opname == 'PUSH_EXC_INFO':
            prefix_blocks.append(inst)
    for inst, o in zip(prefix_blocks, meta.prefix_block_target_offset_remap):
        block_target_offset_remap[cast(int, inst.offset)] = o
    old_start_offset = cast(int, prefix_blocks[-1].offset) if prefix_blocks else -1
    old_inst_offsets = sorted((n for n in setup_fn_target_offsets if n > old_start_offset))
    targets = _filter_iter(instructions, old_inst_offsets, lambda inst, o: inst.offset == o)
    new_targets = _filter_iter(zip(reversed(instructions), reversed(meta.instructions)), targets, lambda v1, v2: v1[0] is v2)
    for new, old in zip(new_targets, targets):
        block_target_offset_remap[old.offset] = new[1].offset