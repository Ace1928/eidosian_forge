from __future__ import annotations
import abc
import argparse
import gzip
import os
import sys
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import hashlib
import typing as T
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from mesonbuild.environment import Environment, detect_ninja
from mesonbuild.mesonlib import (MesonException, RealPathAction, get_meson_command, quiet_git,
from mesonbuild.msetup import add_arguments as msetup_argparse
from mesonbuild.wrap import wrap
from mesonbuild import mlog, build, coredata
from .scripts.meson_exe import run_exe
def process_submodules(self, src: str, distdir: str) -> None:
    module_file = os.path.join(src, '.gitmodules')
    if not os.path.exists(module_file):
        return
    cmd = ['git', 'submodule', 'status', '--cached', '--recursive']
    modlist = subprocess.check_output(cmd, cwd=src, universal_newlines=True).splitlines()
    for submodule in modlist:
        status = submodule[:1]
        sha1, rest = submodule[1:].split(' ', 1)
        subpath = rest.rsplit(' ', 1)[0]
        if status == '-':
            mlog.warning(f'Submodule {subpath!r} is not checked out and cannot be added to the dist')
            continue
        elif status in {'+', 'U'}:
            handle_dirty_opt(f'Submodule {subpath!r} has uncommitted changes that will not be included in the dist tarball', self.options.allow_dirty)
        self.copy_git(os.path.join(src, subpath), distdir, revision=sha1, prefix=subpath)