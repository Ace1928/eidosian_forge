import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def create_call_function(nargs, push_null) -> List[Instruction]:
    """
    Creates a sequence of instructions that makes a function call.

    `push_null` is used in Python 3.11+ only. It is used in codegen when
    a function call is intended to be made with the NULL + fn convention,
    and we know that the NULL has not been pushed yet. We will push a
    NULL and rotate it to the correct position immediately before making
    the function call.
    push_null should default to True unless you know you are calling a function
    that you codegen'd with a null already pushed, for example
    (assume `math` is available in the global scope),

    create_load_global("math", True)  # pushes a null
    create_instruction("LOAD_ATTR", argval="sqrt")
    create_instruction("LOAD_CONST", argval=25)
    create_call_function(1, False)
    """
    if sys.version_info >= (3, 11):
        output = []
        if push_null:
            output.append(create_instruction('PUSH_NULL'))
            output.extend(create_rot_n(nargs + 2))
        output.append(create_instruction('PRECALL', arg=nargs))
        output.append(create_instruction('CALL', arg=nargs))
        return output
    return [create_instruction('CALL_FUNCTION', arg=nargs)]