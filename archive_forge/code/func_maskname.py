import sys
import threading
import os
import select
import struct
import fcntl
import errno
import termios
import array
import logging
import atexit
from collections import deque
from datetime import datetime, timedelta
import time
import re
import asyncore
import glob
import locale
import subprocess
def maskname(mask):
    """
        Returns the event name associated to mask. IN_ISDIR is appended to
        the result when appropriate. Note: only one event is returned, because
        only one event can be raised at a given time.

        @param mask: mask.
        @type mask: int
        @return: event name.
        @rtype: str
        """
    ms = mask
    name = '%s'
    if mask & IN_ISDIR:
        ms = mask - IN_ISDIR
        name = '%s|IN_ISDIR'
    return name % EventsCodes.ALL_VALUES[ms]