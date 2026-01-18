import os
import platform
import sys
from functools import partial
import zmq
from packaging.version import Version as V
from traitlets.config.application import Application
@loop_tk.exit
def loop_tk_exit(kernel):
    """Exit the tk loop."""
    try:
        kernel.app_wrapper.app.destroy()
        del kernel.app_wrapper
    except (RuntimeError, AttributeError):
        pass