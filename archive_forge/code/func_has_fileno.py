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
def has_fileno(obj):
    if not hasattr(obj, 'fileno'):
        return False
    try:
        obj.fileno()
    except (AttributeError, IOError, io.UnsupportedOperation):
        return False
    return True