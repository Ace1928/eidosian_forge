import enum
import logging
import os
import sys
import typing
import warnings
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import callbacks
def set_python_process_info() -> None:
    'Set information about the Python process in an environment variable.\n\n    See discussion at:\n    %s\n    ' % _REFERENCE_TO_R_SESSIONS
    info = (('current_pid', os.getpid()), ('sys.executable', sys.executable))
    info_string = ':'.join(('%s=%s' % x for x in info))
    os.environ[_PYTHON_SESSION_INITIALIZED] = info_string