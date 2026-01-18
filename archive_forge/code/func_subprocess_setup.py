import errno
import os
import signal
import subprocess
from ... import errors, osutils, trace
from ... import transport as _mod_transport
def subprocess_setup():
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)