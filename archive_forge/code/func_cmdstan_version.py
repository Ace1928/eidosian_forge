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
def cmdstan_version() -> Optional[Tuple[int, ...]]:
    """
    Parses version string out of CmdStan makefile variable CMDSTAN_VERSION,
    returns Tuple(Major, minor).

    If CmdStan installation is not found or cannot parse version from makefile
    logs warning and returns None.  Lenient behavoir required for CI tests,
    per comment:
    https://github.com/stan-dev/cmdstanpy/pull/321#issuecomment-733817554
    """
    try:
        makefile = os.path.join(cmdstan_path(), 'makefile')
    except ValueError as e:
        get_logger().info('No CmdStan installation found.')
        get_logger().debug('%s', e)
        return None
    if not os.path.exists(makefile):
        get_logger().info('CmdStan installation %s missing makefile, cannot get version.', cmdstan_path())
        return None
    with open(makefile, 'r') as fd:
        contents = fd.read()
    start_idx = contents.find('CMDSTAN_VERSION := ')
    if start_idx < 0:
        get_logger().info('Cannot parse version from makefile: %s.', makefile)
        return None
    start_idx += len('CMDSTAN_VERSION := ')
    end_idx = contents.find('\n', start_idx)
    version = contents[start_idx:end_idx]
    splits = version.split('.')
    if len(splits) != 3:
        get_logger().info('Cannot parse version, expected "<major>.<minor>.<patch>", found: "%s".', version)
        return None
    return tuple((int(x) for x in splits[0:2]))