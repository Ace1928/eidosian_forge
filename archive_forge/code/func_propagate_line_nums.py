import bisect
import dataclasses
import dis
import sys
from typing import Any, Set, Union
def propagate_line_nums(instructions):
    """Ensure every instruction has line number set in case some are removed"""
    cur_line_no = None

    def populate_line_num(inst):
        nonlocal cur_line_no
        if inst.starts_line:
            cur_line_no = inst.starts_line
        inst.starts_line = cur_line_no
    for inst in instructions:
        populate_line_num(inst)