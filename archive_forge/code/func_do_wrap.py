from __future__ import annotations
import sys
import warnings
from functools import wraps
from types import ModuleType
from typing import TYPE_CHECKING, ClassVar, TypeVar
import attrs
def do_wrap(fn: Callable[ArgsT, RetT]) -> Callable[ArgsT, RetT]:
    nonlocal thing

    @wraps(fn)
    def wrapper(*args: ArgsT.args, **kwargs: ArgsT.kwargs) -> RetT:
        warn_deprecated(thing, version, instead=instead, issue=issue)
        return fn(*args, **kwargs)
    if thing is None:
        thing = wrapper
    if wrapper.__doc__ is not None:
        doc = wrapper.__doc__
        doc = doc.rstrip()
        doc += '\n\n'
        doc += f'.. deprecated:: {version}\n'
        if instead is not None:
            doc += f'   Use {_stringify(instead)} instead.\n'
        if issue is not None:
            doc += f'   For details, see `issue #{issue} <{_url_for_issue(issue)}>`__.\n'
        doc += '\n'
        wrapper.__doc__ = doc
    return wrapper