import os
import platform
import sys
from functools import partial
import zmq
from packaging.version import Version as V
from traitlets.config.application import Application
@loop_qt.exit
def loop_qt_exit(kernel):
    kernel.app.exit()