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
def force_print(self, *args: str, nested: bool, sep: T.Optional[str]=None, end: T.Optional[str]=None) -> None:
    if self.log_disable_stdout:
        return
    iostr = io.StringIO()
    print(*args, sep=sep, end=end, file=iostr)
    raw = iostr.getvalue()
    if self.log_depth:
        prepend = self.log_depth[-1] + '| ' if nested else ''
        lines = []
        for l in raw.split('\n'):
            l = l.strip()
            lines.append(prepend + l if l else '')
        raw = '\n'.join(lines)
    try:
        output = self.log_pager.stdin if self.log_pager else None
        print(raw, end='', file=output)
    except UnicodeEncodeError:
        cleaned = raw.encode('ascii', 'replace').decode('ascii')
        print(cleaned, end='')