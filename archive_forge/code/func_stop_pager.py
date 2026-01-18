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
def stop_pager(self) -> None:
    if self.log_pager:
        try:
            self.log_pager.stdin.flush()
            self.log_pager.stdin.close()
        except OSError:
            pass
        self.log_pager.wait()
        self.log_pager = None