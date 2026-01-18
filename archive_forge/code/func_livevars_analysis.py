import bisect
import dataclasses
import dis
import sys
from typing import Any, Set, Union
def livevars_analysis(instructions, instruction):
    indexof = get_indexof(instructions)
    must = ReadsWrites(set(), set(), set())
    may = ReadsWrites(set(), set(), set())

    def walk(state, start):
        if start in state.visited:
            return
        state.visited.add(start)
        for i in range(start, len(instructions)):
            inst = instructions[i]
            if inst.opcode in HASLOCAL or inst.opcode in HASFREE:
                if 'LOAD' in inst.opname or 'DELETE' in inst.opname:
                    if inst.argval not in must.writes:
                        state.reads.add(inst.argval)
                elif 'STORE' in inst.opname:
                    state.writes.add(inst.argval)
                elif inst.opname == 'MAKE_CELL':
                    pass
                else:
                    raise NotImplementedError(f'unhandled {inst.opname}')
            if inst.exn_tab_entry:
                walk(may, indexof[inst.exn_tab_entry.target])
            if inst.opcode in JUMP_OPCODES:
                walk(may, indexof[inst.target])
                state = may
            if inst.opcode in TERMINAL_OPCODES:
                return
    walk(must, indexof[instruction])
    return must.reads | may.reads