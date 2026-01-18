from __future__ import annotations
from dataclasses import dataclass, InitVar
import os, subprocess
import argparse
import asyncio
import threading
import copy
import shutil
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
import typing as T
import tarfile
import zipfile
from . import mlog
from .ast import IntrospectionInterpreter
from .mesonlib import quiet_git, GitException, Popen_safe, MesonException, windows_proof_rmtree
from .wrap.wrap import (Resolver, WrapException, ALL_TYPES,
def git_reset(self, revision: str) -> bool:
    try:
        self.git_stash()
        self.git_output(['reset', '--hard', 'FETCH_HEAD'])
        self.wrap_resolver.apply_patch(self.wrap.name)
        self.wrap_resolver.apply_diff_files()
    except GitException as e:
        self.log('  -> Could not reset', mlog.bold(self.repo_dir), 'to', mlog.bold(revision))
        self.log(mlog.red(e.output))
        self.log(mlog.red(str(e)))
        return False
    return True