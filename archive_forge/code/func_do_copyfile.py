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
def do_copyfile(self, from_file: str, to_file: str, makedirs: T.Optional[T.Tuple[T.Any, str]]=None, follow_symlinks: T.Optional[bool]=None) -> bool:
    outdir = os.path.split(to_file)[0]
    if not os.path.isfile(from_file) and (not os.path.islink(from_file)):
        raise MesonException(f"Tried to install something that isn't a file: {from_file!r}")
    if os.path.exists(to_file):
        if not os.path.isfile(to_file):
            raise MesonException(f'Destination {to_file!r} already exists and is not a file')
        if self.should_preserve_existing_file(from_file, to_file):
            append_to_log(self.lf, f'# Preserving old file {to_file}\n')
            self.preserved_file_count += 1
            return False
        self.log(f'Installing {from_file} to {outdir}')
        self.remove(to_file)
    else:
        self.log(f'Installing {from_file} to {outdir}')
        if makedirs:
            dirmaker, outdir = makedirs
            dirmaker.makedirs(outdir, exist_ok=True)
    if os.path.islink(from_file):
        if not os.path.exists(from_file):
            self.copy(from_file, outdir, follow_symlinks=False)
        else:
            if follow_symlinks is None:
                follow_symlinks = True
                print(symlink_warning)
            self.copy2(from_file, to_file, follow_symlinks=follow_symlinks)
    else:
        self.copy2(from_file, to_file)
    selinux_updates.append(to_file)
    append_to_log(self.lf, to_file)
    return True