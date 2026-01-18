import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def clean_and_assemble_instructions(instructions: List[Instruction], keys: List[str], code_options: Dict[str, Any]) -> Tuple[List[Instruction], types.CodeType]:
    check_inst_exn_tab_entries_valid(instructions)
    code_options['co_nlocals'] = len(code_options['co_varnames'])
    varname_from_oparg = None
    if sys.version_info >= (3, 11):
        tmp_code = types.CodeType(*[code_options[k] for k in keys])
        varname_from_oparg = tmp_code._varname_from_oparg
    fix_vars(instructions, code_options, varname_from_oparg=varname_from_oparg)
    dirty = True
    while dirty:
        update_offsets(instructions)
        devirtualize_jumps(instructions)
        dirty = bool(fix_extended_args(instructions))
    remove_extra_line_nums(instructions)
    bytecode, lnotab = assemble(instructions, code_options['co_firstlineno'])
    if sys.version_info < (3, 10):
        code_options['co_lnotab'] = lnotab
    else:
        code_options['co_linetable'] = lnotab
    code_options['co_code'] = bytecode
    code_options['co_stacksize'] = stacksize_analysis(instructions)
    assert set(keys) - {'co_posonlyargcount'} == set(code_options.keys()) - {'co_posonlyargcount'}
    if sys.version_info >= (3, 11):
        code_options['co_exceptiontable'] = assemble_exception_table(compute_exception_table(instructions))
    return (instructions, types.CodeType(*[code_options[k] for k in keys]))