import os
import platform
import re
import sys
import types
import warnings
from subprocess import getstatusoutput
def key_value_from_command(cmd, sep, successful_status=(0,), stacklevel=1):
    d = {}
    for line in command_by_line(cmd, successful_status=successful_status, stacklevel=stacklevel + 1):
        l = [s.strip() for s in line.split(sep, 1)]
        if len(l) == 2:
            d[l[0]] = l[1]
    return d