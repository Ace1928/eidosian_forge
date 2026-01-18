import os
import platform
import subprocess
import sys
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple, Union
from tqdm.auto import tqdm
from cmdstanpy import _DOT_CMDSTAN
from .. import progress as progbar
from .logging import get_logger
def get_latest_cmdstan(cmdstan_dir: str) -> Optional[str]:
    """
    Given a valid directory path, find all installed CmdStan versions
    and return highest (i.e., latest) version number.

    Assumes directory consists of CmdStan releases, created by
    function `install_cmdstan`, and therefore dirnames have format
    "cmdstan-<maj>.<min>.<patch>" or "cmdstan-<maj>.<min>.<patch>-rc<num>",
    which is CmdStan release practice as of v 2.24.
    """
    versions = [name[8:] for name in os.listdir(cmdstan_dir) if os.path.isdir(os.path.join(cmdstan_dir, name)) and name.startswith('cmdstan-')]
    if len(versions) == 0:
        return None
    if len(versions) == 1:
        return 'cmdstan-' + versions[0]
    versions = [v for v in versions if v[0].isdigit() and v.count('.') == 2]
    for i in range(len(versions)):
        if '-rc' in versions[i]:
            comps = versions[i].split('-rc')
            mmp = comps[0].split('.')
            rc_num = comps[1]
            patch = str(int(rc_num) - 100)
            versions[i] = '.'.join([mmp[0], mmp[1], patch])
    versions.sort(key=lambda s: list(map(int, s.split('.'))))
    latest = versions[len(versions) - 1]
    mmp = latest.split('.')
    if int(mmp[2]) < 0:
        rc_num = str(int(mmp[2]) + 100)
        mmp[2] = '0-rc' + rc_num
        latest = '.'.join(mmp)
    return 'cmdstan-' + latest