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
def run_dist_steps(meson_command: T.List[str], unpacked_src_dir: str, builddir: str, installdir: str, ninja_args: T.List[str]) -> int:
    if subprocess.call(meson_command + ['--backend=ninja', unpacked_src_dir, builddir]) != 0:
        print('Running Meson on distribution package failed')
        return 1
    if subprocess.call(ninja_args, cwd=builddir) != 0:
        print('Compiling the distribution package failed')
        return 1
    if subprocess.call(ninja_args + ['test'], cwd=builddir) != 0:
        print('Running unit tests on the distribution package failed')
        return 1
    myenv = os.environ.copy()
    myenv['DESTDIR'] = installdir
    if subprocess.call(ninja_args + ['install'], cwd=builddir, env=myenv) != 0:
        print('Installing the distribution package failed')
        return 1
    return 0