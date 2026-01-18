import enum
import logging
import os
import sys
import typing
import warnings
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import callbacks
def is_r_externally_initialized() -> bool:
    r_status = get_r_session_status()
    return str(r_status['current_pid']) == str(r_status.get('PID'))