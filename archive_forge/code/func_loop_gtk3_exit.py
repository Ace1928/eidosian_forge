import os
import platform
import sys
from functools import partial
import zmq
from packaging.version import Version as V
from traitlets.config.application import Application
@loop_gtk3.exit
def loop_gtk3_exit(kernel):
    """Exit the gtk3 loop."""
    kernel._gtk.stop()