import __future__
import builtins
import importlib._bootstrap
import importlib._bootstrap_external
import importlib.machinery
import importlib.util
import inspect
import io
import os
import pkgutil
import platform
import re
import sys
import sysconfig
import time
import tokenize
import urllib.parse
import warnings
from collections import deque
from reprlib import Repr
from traceback import format_exception_only
def serve_until_quit(self):
    while not self.quit:
        rd, wr, ex = select.select([self.socket.fileno()], [], [], 1)
        if rd:
            self.handle_request()
    self.server_close()