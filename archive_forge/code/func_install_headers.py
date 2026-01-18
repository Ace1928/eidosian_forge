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
def install_headers(self, d: InstallData, dm: DirMaker, destdir: str, fullprefix: str) -> None:
    for t in d.headers:
        if not self.should_install(t):
            continue
        fullfilename = t.path
        fname = os.path.basename(fullfilename)
        outdir = get_destdir_path(destdir, fullprefix, t.install_path)
        outfilename = os.path.join(outdir, fname)
        if self.do_copyfile(fullfilename, outfilename, makedirs=(dm, outdir), follow_symlinks=t.follow_symlinks):
            self.did_install_something = True
        self.set_mode(outfilename, t.install_mode, d.install_umask)