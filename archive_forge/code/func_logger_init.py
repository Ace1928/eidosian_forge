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
def logger_init():
    """Initialize logger instance."""
    log = logging.getLogger('pyinotify')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('[%(asctime)s %(name)s %(levelname)s] %(message)s'))
    log.addHandler(console_handler)
    log.setLevel(20)
    return log