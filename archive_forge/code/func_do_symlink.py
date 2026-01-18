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
def do_symlink(self, target: str, link: str, destdir: str, full_dst_dir: str, allow_missing: bool) -> bool:
    abs_target = target
    if not os.path.isabs(target):
        abs_target = os.path.join(full_dst_dir, target)
    elif not os.path.exists(abs_target) and (not allow_missing):
        abs_target = destdir_join(destdir, abs_target)
    if not os.path.exists(abs_target) and (not allow_missing):
        raise MesonException(f'Tried to install symlink to missing file {abs_target}')
    if os.path.exists(link):
        if not os.path.islink(link):
            raise MesonException(f'Destination {link!r} already exists and is not a symlink')
        self.remove(link)
    if not self.printed_symlink_error:
        self.log(f'Installing symlink pointing to {target} to {link}')
    try:
        self.symlink(target, link, target_is_directory=os.path.isdir(abs_target))
    except (NotImplementedError, OSError):
        if not self.printed_symlink_error:
            print('Symlink creation does not work on this platform. Skipping all symlinking.')
            self.printed_symlink_error = True
        return False
    append_to_log(self.lf, link)
    return True