from __future__ import annotations
import os
import errno
import shutil
import subprocess
import sys
from pathlib import Path
from ._backend import Backend
from string import Template
from itertools import chain
import warnings
def run_meson(self, build_dir: Path):
    setup_command = ['meson', 'setup', self.meson_build_dir]
    self._run_subprocess_command(setup_command, build_dir)
    compile_command = ['meson', 'compile', '-C', self.meson_build_dir]
    self._run_subprocess_command(compile_command, build_dir)