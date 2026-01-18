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
def stop_display(self):
    """Closes the display if started"""
    if self._display is not None:
        from .. import logging
        self._display.stop()
        logging.getLogger('nipype.interface').debug('Closing display (if virtual)')