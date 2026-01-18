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
def enable_resource_monitor(self):
    """Sets the resource monitor on"""
    self.resource_monitor = True