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
def cmd_ci_include(self, file: str) -> None:
    self._debug_log_cmd('ci_include', [file])