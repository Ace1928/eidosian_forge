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
def get_fd(self):
    """
        Return assigned inotify's file descriptor.

        @return: File descriptor.
        @rtype: int
        """
    return self._fd