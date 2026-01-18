import codecs
import io
import os
import sys
import warnings
from ..lazy_import import lazy_import
import time
from breezy import (
from .. import config, osutils, trace
from . import NullProgressView, UIFactory
def get_non_echoed_password(self):
    isatty = getattr(self.stdin, 'isatty', None)
    if isatty is not None and isatty():
        import getpass
        password = getpass.getpass('')
    else:
        password = self.stdin.readline()
        if not password:
            password = None
        elif password[-1] == '\n':
            password = password[:-1]
    return password