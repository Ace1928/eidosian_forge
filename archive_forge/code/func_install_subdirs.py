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
def install_subdirs(self, d: InstallData, dm: DirMaker, destdir: str, fullprefix: str) -> None:
    for i in d.install_subdirs:
        if not self.should_install(i):
            continue
        self.did_install_something = True
        full_dst_dir = get_destdir_path(destdir, fullprefix, i.install_path)
        self.log(f'Installing subdir {i.path} to {full_dst_dir}')
        dm.makedirs(full_dst_dir, exist_ok=True)
        self.do_copydir(d, i.path, full_dst_dir, i.exclude, i.install_mode, dm, follow_symlinks=i.follow_symlinks)