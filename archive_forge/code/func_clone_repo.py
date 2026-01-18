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
def clone_repo(repo, path=None, absl=False, add_to_syspath=True):
    path = path or ('/content' if LazyEnv.is_colab else File.curdir())
    if isinstance(repo, str):
        repo = repo.split(',')
    assert isinstance(repo, list), f'Repo must be a list or string: {type(repo)}'
    logger.info(f'Cloning Repo(s): {repo} into {path}')
    for r in repo:
        if 'github.com' not in r:
            r = f'https://github.com/{r}'
        clonepath = File.join(path, File.base(r)) if not absl else path
        try:
            run_cmd(f'git clone {r} {clonepath}')
        except Exception as e:
            logger.error(f'Error Cloning {r}: {str(e)}')
        if add_to_syspath:
            sys.path.append(clonepath)