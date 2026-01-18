from __future__ import annotations
import argparse, datetime, glob, json, os, platform, shutil, sys, tempfile, time
import cProfile as profile
from pathlib import Path
import typing as T
from . import build, coredata, environment, interpreter, mesonlib, mintro, mlog
from .mesonlib import MesonException
def has_build_file(self, dirname: str) -> bool:
    fname = os.path.join(dirname, environment.build_filename)
    return os.path.exists(fname)