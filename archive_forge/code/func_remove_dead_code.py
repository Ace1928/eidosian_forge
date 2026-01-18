import bisect
import dataclasses
import dis
import sys
from typing import Any, Set, Union
def remove_dead_code(instructions):
    """Dead code elimination"""
    indexof = get_indexof(instructions)
    live_code = set()

    def find_live_code(start):
        for i in range(start, len(instructions)):
            if i in live_code:
                return
            live_code.add(i)
            inst = instructions[i]
            if inst.exn_tab_entry:
                find_live_code(indexof[inst.exn_tab_entry.target])
            if inst.opcode in JUMP_OPCODES:
                find_live_code(indexof[inst.target])
            if inst.opcode in TERMINAL_OPCODES:
                return
    find_live_code(0)
    if sys.version_info >= (3, 11):
        live_idx = sorted(live_code)
        for i, inst in enumerate(instructions):
            if i in live_code and inst.exn_tab_entry:
                start_idx = bisect.bisect_left(live_idx, indexof[inst.exn_tab_entry.start])
                assert start_idx < len(live_idx)
                end_idx = bisect.bisect_right(live_idx, indexof[inst.exn_tab_entry.end]) - 1
                assert end_idx >= 0
                assert live_idx[start_idx] <= i <= live_idx[end_idx]
                inst.exn_tab_entry.start = instructions[live_idx[start_idx]]
                inst.exn_tab_entry.end = instructions[live_idx[end_idx]]
    return [inst for i, inst in enumerate(instructions) if i in live_code]