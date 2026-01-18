import os
import sys
import re
from collections.abc import Iterator
from warnings import warn
from looseversion import LooseVersion
import numpy as np
import textwrap
def rgetcwd(error=True):
    """
    Robust replacement for getcwd when folders get removed
    If error==True, this is just an alias for os.getcwd()
    """
    if error:
        return os.getcwd()
    try:
        cwd = os.getcwd()
    except OSError as exc:
        cwd = os.getenv('PWD')
        if cwd is None:
            raise OSError((exc.errno, 'Current directory does not exist anymore, and nipype was not able to guess it from the environment'))
        warn('Current folder does not exist, replacing with "%s" instead.' % cwd)
    return cwd