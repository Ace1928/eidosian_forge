import os
import platform
import sys
from functools import partial
import zmq
from packaging.version import Version as V
from traitlets.config.application import Application
def exit_decorator(exit_func):
    """@func.exit is now a decorator

            to register a function to be called on exit
            """
    func.exit_hook = exit_func
    return exit_func