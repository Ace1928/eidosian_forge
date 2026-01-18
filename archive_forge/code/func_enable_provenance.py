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
def enable_provenance(self):
    """Sets provenance storing on"""
    self._config.set('execution', 'write_provenance', 'true')
    self._config.set('execution', 'hash_method', 'content')