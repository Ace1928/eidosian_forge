import os
import platform
import sys
from functools import partial
import zmq
from packaging.version import Version as V
from traitlets.config.application import Application
def exit_loop():
    """fall back to main loop"""
    app.tk.deletefilehandler(kernel.shell_stream.getsockopt(zmq.FD))
    app.quit()
    app.destroy()
    del kernel.app_wrapper