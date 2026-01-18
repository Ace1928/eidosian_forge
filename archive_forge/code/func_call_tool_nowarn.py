from __future__ import annotations
import typing as T
import os, sys
from .. import mesonlib
from .. import mlog
from ..mesonlib import Popen_safe
import argparse
def call_tool_nowarn(tool: T.List[str], **kwargs: T.Any) -> T.Tuple[str, str]:
    try:
        p, output, e = Popen_safe(tool, **kwargs)
    except FileNotFoundError:
        return (None, '{!r} not found\n'.format(tool[0]))
    except PermissionError:
        return (None, '{!r} not usable\n'.format(tool[0]))
    if p.returncode != 0:
        return (None, e)
    return (output, None)