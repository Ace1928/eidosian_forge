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
def report_bug(exc_info, err_file):
    """Report an exception that probably indicates a bug in brz"""
    from breezy.crash import report_bug
    report_bug(exc_info, err_file)