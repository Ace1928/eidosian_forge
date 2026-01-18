import os
import platform
import sys
from functools import partial
import zmq
from packaging.version import Version as V
from traitlets.config.application import Application
@loop_cocoa.exit
def loop_cocoa_exit(kernel):
    """Exit the cocoa loop."""
    from ._eventloop_macos import stop
    stop()