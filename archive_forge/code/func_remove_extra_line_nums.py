import bisect
import dataclasses
import dis
import sys
from typing import Any, Set, Union
def remove_extra_line_nums(instructions):
    """Remove extra starts line properties before packing bytecode"""
    cur_line_no = None

    def remove_line_num(inst):
        nonlocal cur_line_no
        if inst.starts_line is None:
            return
        elif inst.starts_line == cur_line_no:
            inst.starts_line = None
        else:
            cur_line_no = inst.starts_line
    for inst in instructions:
        remove_line_num(inst)