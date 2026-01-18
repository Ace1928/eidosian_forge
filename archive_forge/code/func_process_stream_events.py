import os
import platform
import sys
from functools import partial
import zmq
from packaging.version import Version as V
from traitlets.config.application import Application
def process_stream_events(*a, **kw):
    """fall back to main loop when there's a socket event"""
    if kernel.shell_stream.flush(limit=1):
        exit_loop()