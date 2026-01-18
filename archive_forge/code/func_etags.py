from __future__ import annotations
import os
import subprocess
from pathlib import Path
import typing as T
def etags() -> int:
    ls = ls_as_bytestream()
    return subprocess.run(['etags', '-'], input=ls).returncode