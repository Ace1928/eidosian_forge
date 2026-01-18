import os
import sys
import errno
import atexit
from warnings import warn
from looseversion import LooseVersion
import configparser
import numpy as np
from simplejson import load, dump
from .misc import str2bool
from filelock import SoftFileLock
@atexit.register
def free_display():
    """Stop virtual display (if it is up)"""
    from .. import config
    config.stop_display()