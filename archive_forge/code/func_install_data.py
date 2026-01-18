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
def install_data(self, d: InstallData, dm: DirMaker, destdir: str, fullprefix: str) -> None:
    for i in d.data:
        if not self.should_install(i):
            continue
        fullfilename = i.path
        outfilename = get_destdir_path(destdir, fullprefix, i.install_path)
        outdir = os.path.dirname(outfilename)
        if self.do_copyfile(fullfilename, outfilename, makedirs=(dm, outdir), follow_symlinks=i.follow_symlinks):
            self.did_install_something = True
        self.set_mode(outfilename, i.install_mode, d.install_umask)