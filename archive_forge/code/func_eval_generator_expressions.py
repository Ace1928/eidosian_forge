from __future__ import annotations
from .. import mesonlib
from .. import mlog
from .common import cmake_is_debug
import typing as T
def eval_generator_expressions() -> str:
    nonlocal i
    i += 2
    func = ''
    args = ''
    res = ''
    exp = ''
    while i < len(raw):
        if raw[i] == '>':
            break
        elif i < len(raw) - 1 and raw[i] == '$' and (raw[i + 1] == '<'):
            exp += eval_generator_expressions()
        else:
            exp += raw[i]
        i += 1
    col_pos = exp.find(':')
    if col_pos < 0:
        func = exp
    else:
        func = exp[:col_pos]
        args = exp[col_pos + 1:]
    func = func.strip()
    args = args.strip()
    if func in supported:
        res = supported[func](args)
    return res