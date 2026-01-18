from __future__ import annotations
from .. import mesonlib
from .. import mlog
from .common import cmake_is_debug
import typing as T
def target_property(arg: str) -> str:
    if ',' not in arg:
        if context_tgt is None:
            return ''
        return ';'.join(context_tgt.properties.get(arg, []))
    args = arg.split(',')
    props = trace.targets[args[0]].properties.get(args[1], []) if args[0] in trace.targets else []
    return ';'.join(props)