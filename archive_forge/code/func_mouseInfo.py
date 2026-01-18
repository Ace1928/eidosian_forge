from __future__ import absolute_import, division, print_function
import collections
import sys
import time
import datetime
import os
import platform
import re
import functools
from contextlib import contextmanager
def mouseInfo():
    """
        This function raises PyAutoGUIException. It's used for the MouseInfo function names if the MouseInfo module
        failed to be imported.
        """
    raise PyAutoGUIException('PyAutoGUI was unable to import mouseinfo. Please install this module to enable the function you tried to call.')