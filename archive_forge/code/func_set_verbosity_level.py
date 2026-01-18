from both of those two places to another location.
import errno
import logging
import os
import sys
import time
from io import StringIO
import breezy
from .lazy_import import lazy_import
from breezy import (
from . import errors
def set_verbosity_level(level):
    """Set the verbosity level.

    :param level: -ve for quiet, 0 for normal, +ve for verbose
    """
    global _verbosity_level
    _verbosity_level = level
    _update_logging_level(level < 0)
    ui.ui_factory.be_quiet(level < 0)