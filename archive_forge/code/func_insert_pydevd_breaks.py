from collections import namedtuple
import dis
from functools import partial
import itertools
import os.path
import sys
from _pydevd_frame_eval.vendored import bytecode
from _pydevd_frame_eval.vendored.bytecode.instr import Instr, Label
from _pydev_bundle import pydev_log
from _pydevd_frame_eval.pydevd_frame_tracing import _pydev_stop_at_break, _pydev_needs_stop_at_break
def insert_pydevd_breaks(code_to_modify, breakpoint_lines, code_line_info=None, _pydev_stop_at_break=_pydev_stop_at_break, _pydev_needs_stop_at_break=_pydev_needs_stop_at_break):
    """
    Inserts pydevd programmatic breaks into the code (at the given lines).

    :param breakpoint_lines: set with the lines where we should add breakpoints.
    :return: tuple(boolean flag whether insertion was successful, modified code).
    """
    if code_line_info is None:
        code_line_info = _get_code_line_info(code_to_modify)
    if not code_line_info.line_to_offset:
        return (False, code_to_modify)
    breakpoint_lines = set(breakpoint_lines)
    for line in breakpoint_lines:
        if line <= 0:
            pydev_log.info('Trying to add breakpoint in invalid line: %s', line)
            return (False, code_to_modify)
    try:
        b = bytecode.Bytecode.from_code(code_to_modify)
        if DEBUG:
            op_number_bytecode = debug_helper.write_bytecode(b, prefix='bytecode.')
        helper_list = _HelperBytecodeList(b)
        modified_breakpoint_lines = breakpoint_lines.copy()
        curr_node = helper_list.head
        added_breaks_in_lines = set()
        last_lineno = None
        while curr_node is not None:
            instruction = curr_node.data
            instruction_lineno = getattr(instruction, 'lineno', None)
            curr_name = getattr(instruction, 'name', None)
            if FIX_PREDICT:
                predict_targets = _PREDICT_TABLE.get(curr_name)
                if predict_targets:
                    next_instruction = curr_node.next.data
                    next_name = getattr(next_instruction, 'name', None)
                    if next_name in predict_targets:
                        next_instruction_lineno = getattr(next_instruction, 'lineno', None)
                        if next_instruction_lineno:
                            next_instruction.lineno = None
            if instruction_lineno is not None:
                if TRACK_MULTIPLE_BRANCHES:
                    if last_lineno is None:
                        last_lineno = instruction_lineno
                    else:
                        if last_lineno == instruction_lineno:
                            if curr_node.prev.data.__class__ != Label:
                                curr_node = curr_node.next
                                continue
                        last_lineno = instruction_lineno
                elif instruction_lineno in added_breaks_in_lines:
                    curr_node = curr_node.next
                    continue
                if instruction_lineno in modified_breakpoint_lines:
                    added_breaks_in_lines.add(instruction_lineno)
                    if curr_node.prev is not None and curr_node.prev.data.__class__ == Label and (curr_name == 'POP_TOP'):
                        for new_instruction in get_instructions_to_add(instruction_lineno, _pydev_stop_at_break=_pydev_stop_at_break, _pydev_needs_stop_at_break=_pydev_needs_stop_at_break):
                            curr_node = curr_node.append(new_instruction)
                    else:
                        for new_instruction in get_instructions_to_add(instruction_lineno, _pydev_stop_at_break=_pydev_stop_at_break, _pydev_needs_stop_at_break=_pydev_needs_stop_at_break):
                            curr_node.prepend(new_instruction)
            curr_node = curr_node.next
        b[:] = helper_list
        if DEBUG:
            debug_helper.write_bytecode(b, op_number_bytecode, prefix='bytecode.')
        new_code = b.to_code()
    except:
        pydev_log.exception('Error inserting pydevd breaks.')
        return (False, code_to_modify)
    if DEBUG:
        op_number = debug_helper.write_dis(code_to_modify)
        debug_helper.write_dis(new_code, op_number)
    return (True, new_code)