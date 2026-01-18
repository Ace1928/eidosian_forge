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
def cmdstan_version_before(major: int, minor: int, info: Optional[Dict[str, str]]=None) -> bool:
    """
    Check that CmdStan version is less than Major.minor version.

    :param major: Major version number
    :param minor: Minor version number

    :return: True if version at or above major.minor, else False.
    """
    cur_version = None
    if info is None or 'stan_version_major' not in info:
        cur_version = cmdstan_version()
    else:
        cur_version = (int(info['stan_version_major']), int(info['stan_version_minor']))
    if cur_version is None:
        get_logger().info('Cannot determine whether version is before %d.%d.', major, minor)
        return False
    if cur_version[0] < major or (cur_version[0] == major and cur_version[1] < minor):
        return True
    return False