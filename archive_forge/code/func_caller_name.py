from __future__ import annotations
import inspect
import os
from inspect import currentframe, getframeinfo
from typing import TYPE_CHECKING
def caller_name(skip: Literal[1, 2]=2) -> str:
    """
    Get a name of a caller in the format module.class.method

    `skip` specifies how many levels of stack to skip while getting caller
    name. skip=1 means "who calls me", skip=2 "who calls my caller" etc.

    An empty string is returned if skipped levels exceed stack height

    Taken from:

        https://gist.github.com/techtonik/2151727

    Public Domain, i.e. feel free to copy/paste
    """
    stack = inspect.stack()
    start = 0 + skip
    if len(stack) < start + 1:
        return ''
    parentframe = stack[start][0]
    name = []
    if (module := inspect.getmodule(parentframe)):
        name.append(module.__name__)
    if 'self' in parentframe.f_locals:
        name.append(parentframe.f_locals['self'].__class__.__name__)
    codename = parentframe.f_code.co_name
    if codename != '<module>':
        name.append(codename)
    del parentframe
    return '.'.join(name)