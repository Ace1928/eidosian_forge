import os
import io
import re
import email.utils
import socket
import sys
import time
import traceback as traceback_
import logging
import platform
import queue
import contextlib
import threading
import urllib.parse
from functools import lru_cache
from . import connections, errors, __version__
from ._compat import bton
from ._compat import IS_PPC
from .workers import threadpool
from .makefile import MakeFile, StreamWriter
def ensure_headers_sent(self):
    """Ensure headers are sent to the client if not already sent."""
    if not self.sent_headers:
        self.sent_headers = True
        self.send_headers()