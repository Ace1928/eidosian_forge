import subprocess
import shlex
import sys
import os
from IPython.utils import py3compat
def read_no_interrupt(p):
    """Read from a pipe ignoring EINTR errors.

    This is necessary because when reading from pipes with GUI event loops
    running in the background, often interrupts are raised that stop the
    command from completing."""
    import errno
    try:
        return p.read()
    except IOError as err:
        if err.errno != errno.EINTR:
            raise