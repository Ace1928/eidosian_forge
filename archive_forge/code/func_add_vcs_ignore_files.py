from __future__ import annotations
import argparse, datetime, glob, json, os, platform, shutil, sys, tempfile, time
import cProfile as profile
from pathlib import Path
import typing as T
from . import build, coredata, environment, interpreter, mesonlib, mintro, mlog
from .mesonlib import MesonException
def add_vcs_ignore_files(self, build_dir: str) -> None:
    with open(os.path.join(build_dir, '.gitignore'), 'w', encoding='utf-8') as ofile:
        ofile.write(git_ignore_file)
    with open(os.path.join(build_dir, '.hgignore'), 'w', encoding='utf-8') as ofile:
        ofile.write(hg_ignore_file)