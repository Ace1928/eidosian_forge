from contextlib import contextmanager
import errno
from io import BytesIO
import json
import mimetypes
from pathlib import Path
import random
import sys
import signal
import threading
import tornado.web
import tornado.ioloop
import tornado.websocket
import matplotlib as mpl
from matplotlib.backend_bases import _Backend
from matplotlib._pylab_helpers import Gcf
from . import backend_webagg_core as core
from .backend_webagg_core import (  # noqa: F401 # pylint: disable=W0611
@contextmanager
def catch_sigint():
    old_handler = signal.signal(signal.SIGINT, lambda sig, frame: ioloop.add_callback_from_signal(shutdown))
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old_handler)