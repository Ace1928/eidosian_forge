from __future__ import annotations
import sys
import os
import stat
import struct
import shutil
import subprocess
import typing as T
from ..mesonlib import OrderedSet, generate_list, Popen_safe
def fix_jar(fname: str) -> None:
    subprocess.check_call(['jar', 'xf', fname, 'META-INF/MANIFEST.MF'])
    with open('META-INF/MANIFEST.MF', 'r+', encoding='utf-8') as f:
        lines = f.readlines()
        f.seek(0)
        for line in lines:
            if not line.startswith('Class-Path:'):
                f.write(line)
        f.truncate()
    subprocess.check_call(['jar', 'ufM', fname, 'META-INF/MANIFEST.MF'])