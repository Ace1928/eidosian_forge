from __future__ import annotations
import enum
import os
import io
import sys
import time
import platform
import shlex
import subprocess
import shutil
import typing as T
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
def start_pager(self) -> None:
    if not colorize_console():
        return
    pager_cmd = []
    if 'PAGER' in os.environ:
        pager_cmd = shlex.split(os.environ['PAGER'])
    else:
        less = shutil.which('less')
        if not less and is_windows():
            git = shutil.which('git')
            if git:
                path = Path(git).parents[1] / 'usr' / 'bin'
                less = shutil.which('less', path=str(path))
        if less:
            pager_cmd = [less]
    if not pager_cmd:
        return
    try:
        env = os.environ.copy()
        if 'LESS' not in env:
            env['LESS'] = 'RXF'
        if 'LV' not in env:
            env['LV'] = '-c'
        self.log_pager = subprocess.Popen(pager_cmd, stdin=subprocess.PIPE, text=True, encoding='utf-8', env=env)
    except Exception as e:
        if 'PAGER' in os.environ:
            from .mesonlib import MesonException
            raise MesonException(f'Failed to start pager: {str(e)}')