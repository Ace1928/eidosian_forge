from __future__ import annotations
import typing as T
import os, sys
from .. import mesonlib
from .. import mlog
from ..mesonlib import Popen_safe
import argparse
def print_tool_warning(tools: T.List[str], msg: str, stderr: T.Optional[str]=None) -> None:
    if os.path.exists(TOOL_WARNING_FILE):
        return
    m = f'{tools!r} {msg}. {RELINKING_WARNING}'
    if stderr:
        m += '\n' + stderr
    mlog.warning(m)
    with open(TOOL_WARNING_FILE, 'w', encoding='utf-8'):
        pass