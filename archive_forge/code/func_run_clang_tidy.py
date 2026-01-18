from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
from .run_tool import run_tool
import typing as T
def run_clang_tidy(fname: Path, builddir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(['clang-tidy', '-p', str(builddir), str(fname)])