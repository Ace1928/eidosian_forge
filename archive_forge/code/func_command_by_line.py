import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def command_by_line(cmd, successful_status=(0,), stacklevel=1):
    ok, output = getoutput(cmd, successful_status=successful_status, stacklevel=stacklevel + 1)
    if not ok:
        return
    for line in output.splitlines():
        yield line.strip()