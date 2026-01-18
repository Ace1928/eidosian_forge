import sys
from distutils.log import *  # noqa: F403
from distutils.log import Log as old_Log
from distutils.log import _global_log
from numpy.distutils.misc_util import (red_text, default_text, cyan_text,
def good(self, msg, *args):
    """
        If we log WARN messages, log this message as a 'nice' anti-warn
        message.

        """
    if WARN >= self.threshold:
        if args:
            print(green_text(msg % _fix_args(args)))
        else:
            print(green_text(msg))
        sys.stdout.flush()