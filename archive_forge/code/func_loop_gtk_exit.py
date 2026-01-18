import os
import platform
import sys
from functools import partial
import zmq
from packaging.version import Version as V
from traitlets.config.application import Application
@loop_gtk.exit
def loop_gtk_exit(kernel):
    """Exit the gtk loop."""
    kernel._gtk.stop()