import pkg_resources
import sys
import optparse
from . import bool_optparse
import os
import re
import textwrap
from . import pluginlib
import configparser
import getpass
from logging.config import fileConfig
def quote_first_command_arg(self, arg):
    """
        There's a bug in Windows when running an executable that's
        located inside a path with a space in it.  This method handles
        that case, or on non-Windows systems or an executable with no
        spaces, it just leaves well enough alone.
        """
    if sys.platform != 'win32' or ' ' not in arg:
        return arg
    try:
        import win32api
    except ImportError:
        raise ValueError('The executable %r contains a space, and in order to handle this issue you must have the win32api module installed' % arg)
    arg = win32api.GetShortPathName(arg)
    return arg