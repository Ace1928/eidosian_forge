import os
import subprocess
import sys
from typing import Callable, List, Optional, TextIO
from .filesystem import pushd
from .logging import get_logger
def returncode_msg(retcode: int) -> str:
    """interpret retcode"""
    if retcode < 0:
        sig = -1 * retcode
        return f'terminated by signal {sig}'
    if retcode <= 125:
        return 'error during processing'
    if retcode == 126:
        return ''
    if retcode == 127:
        return 'program not found'
    sig = retcode - 128
    return f'terminated by signal {sig}'