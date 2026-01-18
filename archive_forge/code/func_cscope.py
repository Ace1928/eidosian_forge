from __future__ import annotations
import os
import subprocess
from pathlib import Path
import typing as T
def cscope() -> int:
    ls = b'\n'.join([b'"%s"' % f for f in ls_as_bytestream().split()])
    return subprocess.run(['cscope', '-v', '-b', '-i-'], input=ls).returncode