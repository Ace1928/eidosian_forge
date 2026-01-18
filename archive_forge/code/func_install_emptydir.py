from __future__ import annotations
from glob import glob
import argparse
import errno
import os
import selectors
import shlex
import shutil
import subprocess
import sys
import typing as T
import re
from . import build, environment
from .backend.backends import InstallData
from .mesonlib import (MesonException, Popen_safe, RealPathAction, is_windows,
from .scripts import depfixer, destdir_join
from .scripts.meson_exe import run_exe
def install_emptydir(self, d: InstallData, dm: DirMaker, destdir: str, fullprefix: str) -> None:
    for e in d.emptydir:
        if not self.should_install(e):
            continue
        self.did_install_something = True
        full_dst_dir = get_destdir_path(destdir, fullprefix, e.path)
        self.log(f'Installing new directory {full_dst_dir}')
        if os.path.isfile(full_dst_dir):
            print(f'Tried to create directory {full_dst_dir} but a file of that name already exists.')
            sys.exit(1)
        dm.makedirs(full_dst_dir, exist_ok=True)
        self.set_mode(full_dst_dir, e.install_mode, d.install_umask)