from __future__ import annotations
import logging
from pathlib import Path
from socket import socket
from typing import Callable, Iterator
from uvicorn.config import Config
from uvicorn.supervisors.basereload import BaseReload
def should_restart(self) -> list[Path] | None:
    self.pause()
    for file in self.iter_py_files():
        try:
            mtime = file.stat().st_mtime
        except OSError:
            continue
        old_time = self.mtimes.get(file)
        if old_time is None:
            self.mtimes[file] = mtime
            continue
        elif mtime > old_time:
            return [file]
    return None