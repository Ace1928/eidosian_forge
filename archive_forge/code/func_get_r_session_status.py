import enum
import logging
import os
import sys
import typing
import warnings
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import callbacks
def get_r_session_status(r_session_init=None) -> dict:
    'Return information about the R session, if available.\n\n    Information about the R session being already initialized can be\n    communicated by an environment variable exported by the process that\n    initialized it. See discussion at:\n    %s\n    ' % _REFERENCE_TO_R_SESSIONS
    res = {'current_pid': os.getpid()}
    if r_session_init is None:
        r_session_init = os.environ.get(_R_SESSION_INITIALIZED)
    if r_session_init:
        for item in r_session_init.split(':'):
            try:
                key, value = item.split('=', 1)
            except ValueError:
                warnings.warn('The item %s in %s should be of the form key=value.' % (item, _R_SESSION_INITIALIZED))
            res[key] = value
    return res