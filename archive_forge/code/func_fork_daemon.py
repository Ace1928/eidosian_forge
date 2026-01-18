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
def fork_daemon():
    pid = os.fork()
    if pid == 0:
        os.setsid()
        pid = os.fork()
        if pid == 0:
            os.chdir('/')
            os.umask(18)
        else:
            os._exit(0)
    else:
        os._exit(0)
    fd_inp = os.open(stdin, os.O_RDONLY)
    os.dup2(fd_inp, 0)
    fd_out = os.open(stdout, os.O_WRONLY | os.O_CREAT, 384)
    os.dup2(fd_out, 1)
    fd_err = os.open(stderr, os.O_WRONLY | os.O_CREAT, 384)
    os.dup2(fd_err, 2)