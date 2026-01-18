from __future__ import annotations
import typing as T
import os, sys
from .. import mesonlib
from .. import mlog
from ..mesonlib import Popen_safe
import argparse
def get_tool(name: str) -> T.List[str]:
    evar = name.upper()
    if evar in os.environ:
        import shlex
        return shlex.split(os.environ[evar])
    return [name]