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
@contextmanager
def nested_warnings(self) -> T.Iterator[None]:
    old = self.log_warnings_counter
    self.log_warnings_counter = 0
    try:
        yield
    finally:
        self.log_warnings_counter = old