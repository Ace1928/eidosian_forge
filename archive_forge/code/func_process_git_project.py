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
def process_git_project(self, src_root: str, distdir: str) -> None:
    if self.have_dirty_index():
        handle_dirty_opt(msg_uncommitted_changes, self.options.allow_dirty)
    if os.path.exists(distdir):
        windows_proof_rmtree(distdir)
    repo_root = self.git_root(src_root)
    if repo_root.samefile(src_root):
        os.makedirs(distdir)
        self.copy_git(src_root, distdir)
    else:
        subdir = Path(src_root).relative_to(repo_root)
        tmp_distdir = distdir + '-tmp'
        if os.path.exists(tmp_distdir):
            windows_proof_rmtree(tmp_distdir)
        os.makedirs(tmp_distdir)
        self.copy_git(repo_root, tmp_distdir, subdir=str(subdir))
        Path(tmp_distdir, subdir).rename(distdir)
        windows_proof_rmtree(tmp_distdir)
    self.process_submodules(src_root, distdir)