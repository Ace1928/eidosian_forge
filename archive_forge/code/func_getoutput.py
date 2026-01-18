import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def getoutput(cmd, successful_status=(0,), stacklevel=1):
    try:
        status, output = getstatusoutput(cmd)
    except OSError as e:
        warnings.warn(str(e), UserWarning, stacklevel=stacklevel)
        return (False, '')
    if os.WIFEXITED(status) and os.WEXITSTATUS(status) in successful_status:
        return (True, output)
    return (False, output)