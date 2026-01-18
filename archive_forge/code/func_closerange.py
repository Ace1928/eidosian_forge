import ast
import email.utils
import errno
import fcntl
import html
import importlib
import inspect
import io
import logging
import os
import pwd
import random
import re
import socket
import sys
import textwrap
import time
import traceback
import warnings
from gunicorn.errors import AppImportError
from gunicorn.workers import SUPPORTED_WORKERS
import urllib.parse
def closerange(fd_low, fd_high):
    for fd in range(fd_low, fd_high):
        try:
            os.close(fd)
        except OSError:
            pass