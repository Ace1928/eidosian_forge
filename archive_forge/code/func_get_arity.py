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
def get_arity(f):
    sig = inspect.signature(f)
    arity = 0
    for param in sig.parameters.values():
        if param.kind in positionals:
            arity += 1
    return arity