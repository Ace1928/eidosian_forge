from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import subprocess
import importlib
import pkg_resources
import threading
from subprocess import check_output
from dataclasses import dataclass
from typing import Optional
from fileio import File, PathIO, PathIOLike
from lazyops.envs import logger
from lazyops.envs import LazyEnv
def lazy_install(req, force=False, latest=False, verbose=False):
    req_base = req.split('=')[0].replace('>', '').replace('<', '').strip()
    if lazy_check(req_base) and (not force):
        return
    python = sys.executable
    pip_exec = [python, '-m', 'pip', 'install']
    if '=' not in req or latest:
        pip_exec.append('--upgrade')
    pip_exec.append(req)
    subprocess.check_call(pip_exec, stdout=subprocess.DEVNULL)
    if verbose:
        logger.info(f'{req} installed successfully.')