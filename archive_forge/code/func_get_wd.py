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
def get_wd(self, path):
    """
        Returns the watch descriptor associated to path. This method
        presents a prohibitive cost, always prefer to keep the WD
        returned by add_watch(). If the path is unknown it returns None.

        @param path: Path.
        @type path: str
        @return: WD or None.
        @rtype: int or None
        """
    path = self.__format_path(path)
    for iwd in self._wmd.items():
        if iwd[1].path == path:
            return iwd[0]