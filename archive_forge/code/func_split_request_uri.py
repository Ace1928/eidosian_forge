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
def split_request_uri(uri):
    if uri.startswith('//'):
        parts = urllib.parse.urlsplit('.' + uri)
        return parts._replace(path=parts.path[1:])
    return urllib.parse.urlsplit(uri)