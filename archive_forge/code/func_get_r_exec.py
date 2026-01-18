import argparse
import enum
import logging
import os
import shlex
import subprocess
import sys
from typing import Optional
import warnings
def get_r_exec(r_home: str) -> str:
    """Get the path of the R executable/binary.

    :param: R HOME directory
    :return: Path to the R executable/binary"""
    if sys.platform == 'win32' and '64 bit' in sys.version:
        r_exec = os.path.join(r_home, 'bin', 'x64', 'R')
    else:
        r_exec = os.path.join(r_home, 'bin', 'R')
    logger.info(f'R exec path: {r_exec}')
    return r_exec