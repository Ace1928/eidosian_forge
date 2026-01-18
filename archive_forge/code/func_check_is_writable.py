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
def check_is_writable(path):
    try:
        with open(path, 'a') as f:
            f.close()
    except IOError as e:
        raise RuntimeError("Error: '%s' isn't writable [%r]" % (path, e))