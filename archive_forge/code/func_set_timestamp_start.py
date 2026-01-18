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
def set_timestamp_start(self, start: float) -> None:
    self.log_timestamp_start = start