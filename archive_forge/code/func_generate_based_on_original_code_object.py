import copy
import dataclasses
import sys
import types
from typing import Any, cast, Dict, List, Optional, Tuple
from .bytecode_transformation import (
from .utils import ExactWeakKeyDictionary
@classmethod
def generate_based_on_original_code_object(cls, code, lineno, offset: int, setup_fn_target_offsets: Tuple[int, ...], *args):
    """
        This handles the case of generating a resume into code generated
        to resume something else.  We want to always generate starting
        from the original code object so that if control flow paths
        converge we only generated 1 resume function (rather than 2^n
        resume functions).
        """
    meta: ResumeFunctionMetadata = ContinueExecutionCache.generated_code_metadata[code]
    new_offset = None

    def find_new_offset(instructions: List[Instruction], code_options: Dict[str, Any]):
        nonlocal new_offset
        target, = (i for i in instructions if i.offset == offset)
        new_target, = (i2 for i1, i2 in zip(reversed(instructions), reversed(meta.instructions)) if i1 is target)
        assert target.opcode == new_target.opcode
        new_offset = new_target.offset
    transform_code_object(code, find_new_offset)
    if sys.version_info >= (3, 11):
        if not meta.block_target_offset_remap:
            block_target_offset_remap = meta.block_target_offset_remap = {}

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
            transform_code_object(code, remap_block_offsets)
        setup_fn_target_offsets = tuple((block_target_offset_remap[n] for n in setup_fn_target_offsets))
    return ContinueExecutionCache.lookup(meta.code, lineno, new_offset, setup_fn_target_offsets, *args)