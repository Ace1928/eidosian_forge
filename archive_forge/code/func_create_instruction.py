import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def create_instruction(name, *, arg=None, argval=_NotProvided, target=None) -> Instruction:
    """
    At most one of `arg`, `argval`, and `target` can be not None/_NotProvided.
    This is to prevent ambiguity, e.g. does
        create_instruction("LOAD_CONST", 5)
    mean load the constant at co_consts[5], or load the constant 5?

    If `arg` is not provided, it will be computed during assembly from
    `argval` or `target`.

    Do not use for LOAD_GLOBAL - use create_load_global instead.
    """
    assert name != 'LOAD_GLOBAL'
    cnt = (arg is not None) + (argval is not _NotProvided) + (target is not None)
    if cnt > 1:
        raise RuntimeError('only one of arg, argval, and target can be not None/_NotProvided')
    if arg is not None and (not isinstance(arg, int)):
        raise RuntimeError('instruction arg must be int or None')
    return Instruction(opcode=dis.opmap[name], opname=name, arg=arg, argval=argval, target=target)