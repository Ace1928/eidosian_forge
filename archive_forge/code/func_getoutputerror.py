import subprocess
import shlex
import sys
import os
from IPython.utils import py3compat
def getoutputerror(cmd):
    """Return (standard output, standard error) of executing cmd in a shell.

    Accepts the same arguments as os.system().

    Parameters
    ----------
    cmd : str or list
        A command to be executed in the system shell.

    Returns
    -------
    stdout : str
    stderr : str
    """
    return get_output_error_code(cmd)[:2]